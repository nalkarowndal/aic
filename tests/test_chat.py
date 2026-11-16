import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
from docx import Document
from openai import APIConnectionError
from PIL import Image
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
sys.path.insert(0, str(APP_PATH.parent))


def make_app():
    app = AppTest.from_file(str(APP_PATH), default_timeout=10)
    app.secrets["GROQ_API_KEY"] = "offline-test"
    return app


def chunks(*texts, finish_reason="stop"):
    return iter(
        [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text), finish_reason=None)])
         for text in texts]
        + [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason=finish_reason)])]
    )


def document_bytes(text):
    document = Document()
    document.add_paragraph(text)
    document.add_table(rows=1, cols=1).cell(0, 0).text = "Table detail"
    content = io.BytesIO()
    document.save(content)
    return content.getvalue()


def pdf_bytes(text=None):
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    if text is not None:
        contents = DecodedStreamObject()
        contents.set_data(f"BT /F1 12 Tf 20 200 Td ({text}) Tj ET".encode())
        page.replace_contents(contents)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({
                NameObject("/F1"): DictionaryObject({
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                }),
            }),
        })
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


class ChatAppTest(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.client.__enter__.return_value = self.client
        self.create = self.client.chat.completions.create
        client_patch = patch("openai.OpenAI", return_value=self.client)
        client_patch.start()
        self.addCleanup(client_patch.stop)

    def test_successful_response_stays_visible_and_next_request_keeps_context(self):
        self.create.side_effect = [chunks("First ", "answer"), chunks("Second answer")]
        app = make_app().run()
        app.chat_input[0].set_value("First question").run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("First answer", [m.value for m in app.markdown])

        app.chat_input[0].set_value("Follow-up question").run()
        history = self.create.call_args.kwargs["messages"]
        self.assertEqual(
            [message["content"] for message in history[1:]],
            ["First question", "First answer", "Follow-up question"],
        )

    def test_retry_updates_failed_turn_in_place(self):
        self.create.side_effect = [
            APIConnectionError(request=httpx.Request("POST", "https://api.groq.com")),
            chunks("Recovered answer"),
        ]
        app = make_app().run()
        app.chat_input[0].set_value("Retry this question").run()
        button = next(button for button in app.button if button.label == "다시 시도")
        button.click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual([m.value for m in app.markdown].count("Retry this question"), 1)
        self.assertIn("Recovered answer", [m.value for m in app.markdown])

    def test_new_chat_keeps_previous_conversation_after_reload(self):
        self.create.return_value = chunks("Archived answer")
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"AI_STUDIO_HISTORY_DB": str(Path(directory) / "history.sqlite3")}
        ), patch("conversation_history.browser_owner", return_value="a" * 64):
            app = make_app().run()
            app.chat_input[0].set_value("First conversation").run()
            next(button for button in app.button if button.label == "새 대화").click().run()
            self.assertNotIn("Archived answer", [message.value for message in app.markdown])

            reopened = make_app().run()
            next(button for button in reopened.button if button.label == "First conversation").click().run()
            self.assertIn("Archived answer", [message.value for message in reopened.markdown])

    def test_another_tab_cannot_restore_a_deleted_record(self):
        self.create.side_effect = [chunks("Private answer"), chunks("Another answer")]
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"AI_STUDIO_HISTORY_DB": str(Path(directory) / "history.sqlite3")}
        ), patch("conversation_history.browser_owner", return_value="a" * 64):
            app = make_app().run()
            app.chat_input[0].set_value("Deleted conversation").run()
            other = make_app().run()
            next(button for button in other.button if button.label == "Deleted conversation").click().run()
            next(button for button in app.button if button.label == "현재 대화 삭제").click().run()

            other.chat_input[0].set_value("Old tab still open").run()
            reopened = make_app().run()
            self.assertNotIn("Deleted conversation", [button.label for button in reopened.button])

    def test_settings_use_the_selected_providers_key_and_model(self):
        from openai import OpenAI

        providers = (
            ("OpenAI", "https://api.openai.com/v1", "max_completion_tokens"),
            ("Gemini", "https://generativelanguage.googleapis.com/v1beta/openai/", "max_tokens"),
        )
        for provider, endpoint, token_parameter in providers:
            with self.subTest(provider=provider):
                self.create.return_value = chunks("Provider answer")
                app = make_app().run()
                app.radio(key="settings_provider").set_value(provider).run()
                next(field for field in app.text_input if field.label == "API 키").set_value(f"offline-{provider}-key")
                next(field for field in app.text_input if field.label == "텍스트 모델").set_value("custom-text-model")
                next(widget for widget in app.selectbox if widget.label == "답변 언어").select("한국어")
                next(button for button in app.button if button.label == "설정 적용").click().run()
                app.chat_input[0].set_value("Hello").run()

                self.assertEqual(OpenAI.call_args.kwargs["api_key"], f"offline-{provider}-key")
                self.assertEqual(OpenAI.call_args.kwargs["base_url"], endpoint)
                self.assertEqual(self.create.call_args.kwargs["model"], "custom-text-model")
                self.assertEqual(self.create.call_args.kwargs[token_parameter], 3_000)
                self.assertIn("Korean", self.create.call_args.kwargs["messages"][0]["content"])

    def test_length_limited_answer_continues_in_the_same_turn(self):
        self.create.side_effect = [chunks("Part one.", finish_reason="length"), chunks(" Part two.")]
        app = make_app().run()
        app.chat_input[0].set_value("Write a long answer").run()
        next(button for button in app.button if button.label == "답변 이어쓰기").click().run()
        self.assertIn("Part one. Part two.", [message.value for message in app.markdown])
        self.assertEqual(len(app.session_state["turns"]), 1)
        self.assertIn({"role": "assistant", "content": "Part one."}, self.create.call_args.kwargs["messages"])

    def test_interrupted_network_stream_keeps_the_received_text(self):
        def broken_stream():
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Partial"), finish_reason=None)])
            raise httpx.ReadError("connection closed")

        self.create.return_value = broken_stream()
        app = make_app().run()
        app.chat_input[0].set_value("Wait for the response").run()

        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertIn("Partial", [message.value for message in app.markdown])
        self.assertIn("답변 이어쓰기", [button.label for button in app.button])

    def test_failed_replacement_does_not_send_previous_document(self):
        good = SimpleNamespace(name="old.txt", getvalue=lambda: b"PRIVATE-OLD-TEXT")
        bad = SimpleNamespace(name="new.txt", getvalue=lambda: b"\xff")
        with patch("streamlit.file_uploader", return_value=good) as uploader:
            app = make_app().run()
            uploader.return_value = bad
            app.run()
            self.assertGreater(len(app.error), 0)
            self.assertTrue(app.chat_input[0].proto.disabled)
            self.create.assert_not_called()
            uploader.return_value = None
            app.run()
            self.create.return_value = chunks("No stale attachment")
            app.chat_input[0].set_value("Now send without a file").run()
        request = self.create.call_args.kwargs["messages"]
        self.assertNotIn("PRIVATE-OLD-TEXT", str(request))

    def test_switching_files_hides_the_previous_files_follow_up_action(self):
        first = SimpleNamespace(name="first.txt", getvalue=lambda: b"First source")
        second = SimpleNamespace(name="second.txt", getvalue=lambda: b"Second source")
        self.create.return_value = chunks("Answer [1]")
        with patch("streamlit.file_uploader", return_value=first) as uploader:
            app = make_app().run()
            app.chat_input[0].set_value("Summarize this file").run()
            self.assertIn("followup_0", [button.key for button in app.button])
            uploader.return_value = second
            app.run()
        self.assertNotIn("followup_0", [button.key for button in app.button])


class AttachmentTest(unittest.TestCase):
    def test_citations_resolve_common_brackets_without_showing_unknown_sources(self):
        from chat_service import cited_sources, parse_attachment

        attachment = parse_attachment("sample.txt", b"First source")
        for answer in ("Claim [1].", "Claim 【1】.", "Claim ［１］."):
            with self.subTest(answer=answer):
                matches = cited_sources(attachment, answer + " Unverified [999].")
                self.assertEqual(matches, [(1, attachment.sources[0])])

    def test_pdf_preserves_page_location_and_rejects_scans_without_text(self):
        from chat_service import AttachmentError, parse_attachment

        attachment = parse_attachment("sample.pdf", pdf_bytes("Moonlight Project"))
        self.assertEqual(attachment.sources[0].label, "1쪽")
        self.assertIn("Moonlight Project", attachment.sources[0].text)
        with self.assertRaises(AttachmentError):
            parse_attachment("scan.pdf", pdf_bytes())

    def test_mixed_pdf_reports_pages_skipped_due_to_missing_text(self):
        from chat_service import parse_attachment

        writer = PdfWriter()
        writer.append(io.BytesIO(pdf_bytes("Readable first page")))
        writer.add_blank_page(width=300, height=300)
        output = io.BytesIO()
        writer.write(output)
        attachment = parse_attachment("mixed.pdf", output.getvalue())
        self.assertEqual(attachment.total_pages, 2)
        self.assertEqual(attachment.unreadable_pages, (2,))
        self.assertTrue(attachment.truncated)

    def test_docx_extracts_table_content_and_image_type_is_detected(self):
        from chat_service import parse_attachment

        document = parse_attachment("sample.docx", document_bytes("Document paragraph"))
        self.assertEqual([source.text for source in document.sources], ["Document paragraph", "Table detail"])
        output = io.BytesIO()
        Image.new("RGB", (10, 20), "blue").save(output, format="PNG")
        image = parse_attachment("sample.jpg", output.getvalue())
        self.assertEqual(image.mime_type, "image/png")
        self.assertEqual(image.dimensions, (10, 20))
