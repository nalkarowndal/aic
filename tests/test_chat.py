import io
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


def make_app():
    app = AppTest.from_file(str(APP_PATH), default_timeout=10)
    app.secrets["GROQ_API_KEY"] = "offline-test"
    return app


def chunks(*texts):
    return iter(
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])
        for text in texts
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

    def test_docx_extracts_table_content_and_image_type_is_detected(self):
        from chat_service import parse_attachment

        document = parse_attachment("sample.docx", document_bytes("Document paragraph"))
        self.assertEqual([source.text for source in document.sources], ["Document paragraph", "Table detail"])
        output = io.BytesIO()
        Image.new("RGB", (10, 20), "blue").save(output, format="PNG")
        image = parse_attachment("sample.jpg", output.getvalue())
        self.assertEqual(image.mime_type, "image/png")
        self.assertEqual(image.dimensions, (10, 20))
