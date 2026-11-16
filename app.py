import logging
import os
from hashlib import sha256
from itertools import chain
from pathlib import Path

import streamlit as st
from jinja2 import Environment, FileSystemLoader, select_autoescape
from openai import APIConnectionError, APIStatusError, OpenAI, OpenAIError
from streamlit.errors import StreamlitSecretNotFoundError

from chat_service import (
    FILE_TYPES,
    GROQ_URL,
    MAX_UPLOAD_MB,
    TEXT_MODEL,
    VISION_MODEL,
    AttachmentError,
    Turn,
    build_request,
    cited_sources,
    parse_attachment,
)


ROOT = Path(__file__).resolve().parent
PAGE_WIDTH = 800
LOG = logging.getLogger(__name__)
SUGGESTIONS = {
    "문서 요약": "첨부한 문서를 세 가지 핵심 항목으로 요약해 줘.",
    "문서에서 항목 찾기": "첨부한 문서에서 결정 사항과 그 근거를 찾아 줘.",
    "이미지 설명": "첨부한 이미지에서 보이는 내용을 설명해 줘.",
}

st.set_page_config(page_title="AI Studio", page_icon="✦", layout="wide")
state = st.session_state
state.setdefault("turns", [])
state.setdefault("uploader_version", 0)
state.setdefault("attachment", None)
state.setdefault("attachment_error", None)
state.setdefault("upload_signature", None)


def setting(name, default=""):
    try:
        value = os.environ.get(name) or st.secrets.get(name, default)
    except StreamlitSecretNotFoundError:
        return default
    return str(value).strip()


def sync_attachment(upload):
    data = upload.getvalue() if upload is not None else None
    signature = (upload.name, sha256(data).digest()) if upload is not None else None
    if signature == state.upload_signature:
        return

    state.upload_signature = signature
    state.attachment = None
    state.attachment_error = None
    if signature is None:
        return
    try:
        with st.spinner("파일을 읽고 있어요"):
            state.attachment = parse_attachment(upload.name, data)
    except AttachmentError as exc:
        state.attachment_error = str(exc)


def clear_attachment():
    state.uploader_version += 1
    sync_attachment(None)


def new_chat():
    state.turns = []
    state.question = ""
    clear_attachment()


def set_question(text):
    state.question = text


def attachment_caption(attachment):
    if attachment.image:
        width, height = attachment.dimensions
        return f"이미지 준비 완료 · {width:,} × {height:,} px"
    if attachment.truncated:
        return "문서 일부만 답변에 사용됩니다."
    return "문서에서 텍스트를 읽었습니다."


def render_attachment():
    attachment = state.attachment
    if not attachment and not state.attachment_error:
        return

    with st.container(key="attachment-card", horizontal=True, vertical_alignment="center"):
        with st.container():
            st.text(state.upload_signature[0])
            if state.attachment_error:
                st.error(state.attachment_error)
            else:
                st.caption(attachment_caption(attachment))

        if attachment:
            with st.popover("미리보기", type="tertiary"):
                if attachment.image:
                    st.image(attachment.image, width=280)
                else:
                    st.caption(attachment.sources[0].label)
                    st.text(attachment.sources[0].text[:800])
        st.button("제거", type="tertiary", on_click=clear_attachment)


def request_error(exc):
    if isinstance(exc, APIConnectionError):
        return "Groq에 연결할 수 없습니다. 잠시 후 다시 시도해 주세요."
    if isinstance(exc, APIStatusError):
        if exc.status_code in (401, 403):
            return "Groq 인증 또는 모델 접근 권한을 확인해 주세요."
        if exc.status_code == 429:
            return "Groq 요청 한도에 도달했습니다. 잠시 후 다시 시도해 주세요."
        if exc.status_code in (400, 404, 413):
            return "Groq에서 모델이나 입력을 처리할 수 없었습니다. 첨부 파일과 모델 설정을 확인해 주세요."
    return "답변을 가져오는 중 문제가 생겼습니다. 잠시 후 다시 시도해 주세요."


def generate_answer(turn, history):
    key = setting("GROQ_API_KEY")
    turn.answer = ""
    turn.error = None
    if not key:
        turn.error = "Groq 키가 설정되지 않았습니다. 앱 설정에서 연결을 확인해 주세요."
        return

    model, messages = build_request(
        turn,
        history,
        text_model=setting("GROQ_TEXT_MODEL", TEXT_MODEL) or TEXT_MODEL,
        vision_model=setting("GROQ_VISION_MODEL", VISION_MODEL) or VISION_MODEL,
    )
    parts = []

    try:
        with OpenAI(base_url=GROQ_URL, api_key=key, timeout=35, max_retries=0) as client:
            with st.spinner("답변을 준비하고 있어요"):
                stream = client.chat.completions.create(
                    model=model, messages=messages, stream=True, max_completion_tokens=3_000
                )
                tokens = (
                    chunk.choices[0].delta.content for chunk in stream
                    if chunk.choices and chunk.choices[0].delta.content
                )
                first_token = next(tokens, None)

            if first_token:
                def output():
                    for text in chain((first_token,), tokens):
                        parts.append(text)
                        yield text

                st.write_stream(output())
            else:
                turn.error = "Groq에서 표시할 답변을 받지 못했습니다. 다시 시도해 주세요."
    except OpenAIError as exc:
        LOG.warning("Groq request failed (%s, status %s)", type(exc).__name__, getattr(exc, "status_code", None))
        turn.error = request_error(exc)
    finally:
        turn.answer = "".join(parts)


def render_turn(turn, index, generate=False):
    with st.container(key=f"user-turn-{index}"), st.chat_message("user", avatar=":material/person:"):
        if turn.attachment:
            st.text(f"첨부 · {turn.attachment.name}")
        st.markdown(turn.question)

    with st.container(key=f"assistant-turn-{index}"), st.chat_message("assistant", avatar=":material/auto_awesome:"):
        if generate:
            generate_answer(turn, state.turns[:index])
        elif turn.answer:
            st.markdown(turn.answer)

        if turn.error:
            st.error(turn.error)
            st.button("다시 시도", key=f"retry_{index}", on_click=state.update, args=({"retry": index},))
            return
        if not turn.answer:
            return

        attachment = turn.attachment
        if attachment and attachment.sources:
            if attachment.truncated:
                st.caption("이 답변에는 문서의 일부만 사용되었습니다.")
            references = cited_sources(attachment, turn.answer)
            if references:
                with st.expander(f"문서 근거 {len(references)}개 보기"):
                    for number, source in references:
                        st.text(f"[{number}] {source.label} · {attachment.name}")
                        st.text(source.text)

        with st.container(horizontal=True, key=f"answer-actions-{index}"):
            with st.popover("복사", type="tertiary", icon=":material/content_copy:"):
                st.code(turn.answer, language=None, wrap_lines=True)
            st.download_button(
                "저장", turn.answer, file_name="ai-studio-answer.md", mime="text/markdown",
                key=f"download_{index}", on_click="ignore", type="tertiary", icon=":material/download:",
            )
        if attachment and not attachment.image and index == len(state.turns) - 1 and attachment is state.attachment:
            st.button(
                "근거를 더 자세히 설명", key=f"followup_{index}", type="secondary",
                on_click=set_question, args=("방금 답변에서 사용한 문서 근거를 더 자세히 설명해 줘.",),
            )


st.html(f"<style>{(ROOT / 'styles.css').read_text(encoding='utf-8')}</style>")
templates = Environment(loader=FileSystemLoader(ROOT / "templates"), autoescape=select_autoescape())
with st.container(key="studio-header", width=PAGE_WIDTH, horizontal=True, vertical_alignment="center"):
    st.html(templates.get_template("header.html").render(title="AI Studio"))
    st.button("새 대화", icon=":material/add:", on_click=new_chat, help="현재 대화와 첨부를 비웁니다.")

transcript = st.container(key="studio-transcript", width=PAGE_WIDTH)
welcome = transcript.empty()
if not state.turns:
    with welcome.container():
        with st.container(key="studio-welcome"):
            st.html('<h2 class="studio-welcome-title">문서나 이미지에 대해<br>무엇이 궁금한가요?</h2>')
            st.caption("파일을 첨부하거나 아래 예시로 시작해 보세요.")
            with st.container(horizontal=True, key="studio-suggestions"):
                for label, prompt in SUGGESTIONS.items():
                    st.button(label, on_click=set_question, args=(prompt,))

with st.bottom:
    with st.container(key="studio-composer", width=PAGE_WIDTH):
        attachment_card = st.container()
        row = st.container(key="studio-input-row", horizontal=True, wrap=False, vertical_alignment="center")
        with row.popover("첨부", icon=":material/attach_file:"):
            upload = st.file_uploader(
                "문서 또는 이미지", type=FILE_TYPES, max_upload_size=MAX_UPLOAD_MB,
                key=f"upload_{state.uploader_version}",
            )
            st.caption(f"파일 1개 · 최대 {MAX_UPLOAD_MB}MB")
        with attachment_card:
            sync_attachment(upload)
            render_attachment()
        question = row.chat_input(
            "메시지를 입력하세요", key="question", max_chars=4_000,
            disabled=bool(state.attachment_error), submit_mode="disable",
        )

retry = state.pop("retry", None)
with transcript:
    for index, turn in enumerate(state.turns):
        render_turn(turn, index, generate=index == retry)

if question and question.strip() and not state.attachment_error:
    turn = Turn(question.strip(), state.attachment)
    state.turns.append(turn)
    welcome.empty()
    with transcript:
        render_turn(turn, len(state.turns) - 1, generate=True)
    st.rerun()
