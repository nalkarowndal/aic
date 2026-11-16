import logging
import sqlite3
from hashlib import sha256
from itertools import chain
from pathlib import Path
from uuid import uuid4

import streamlit as st
from httpx import TransportError
from jinja2 import Environment, FileSystemLoader, select_autoescape
from openai import APIConnectionError, APIStatusError, OpenAI, OpenAIError

import conversation_history
from chat_service import (
    FILE_TYPES,
    MAX_UPLOAD_MB,
    Attachment,
    AttachmentError,
    Turn,
    build_request,
    citation_links,
    cited_sources,
    parse_attachment,
)
from chat_settings import LANGUAGES, PROVIDERS, TOKEN_LIMITS, init_settings, model_defaults, render_settings, server_key


ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger(__name__)
SUGGESTIONS = {
    "파일 없이 질문하기": "오늘 할 일을 우선순위에 따라 정리하는 방법을 알려 줘.",
    "문서 요약": "첨부한 문서를 세 가지 핵심 항목으로 요약해 줘.",
    "문서에서 항목 찾기": "첨부한 문서에서 결정 사항과 그 근거를 찾아 줘.",
    "이미지 설명": "첨부한 이미지에서 보이는 내용을 설명해 줘.",
}
LENGTH_INSTRUCTIONS = {"짧게": "Be concise.", "자세히": "Give a thorough answer."}
INCOMPLETE_LABELS = {
    "length": "답변이 길이 제한에 도달했어요.",
    "interrupted": "답변 생성이 중단됐어요.",
    "filtered": "공급자의 콘텐츠 제한으로 답변이 종료됐어요.",
}

st.set_page_config(page_title="AI Studio", page_icon="✦", layout="wide")
state = st.session_state
state.setdefault("turns", [])
state.setdefault("uploader_version", 0)
state.setdefault("attachment", None)
state.setdefault("attachment_error", None)
state.setdefault("upload_signature", None)
state.setdefault("session_chats", {})
state.setdefault("history_notice", None)
state.setdefault("chat_revision", 0)
if "chat_id" not in state:
    state.chat_id = uuid4().hex
    if state.turns:
        state.turns = [
            Turn(turn.question, Attachment(**vars(turn.attachment)) if turn.attachment else None,
                 turn.answer, turn.error, "error" if turn.error else "complete")
            for turn in state.turns
        ]
        state.session_chats[state.chat_id] = (state.turns, 0)
init_settings(state)

owner = conversation_history.browser_owner()
store = None
if owner:
    try:
        store = conversation_history.HistoryStore(conversation_history.history_path())
    except (OSError, sqlite3.Error):
        state.history_notice = "대화 저장소를 열 수 없어 현재 접속에만 기록이 남아요."


def persist_chat():
    if not state.turns:
        return
    if store:
        original_id = state.chat_id
        try:
            revision = store.save(owner, original_id, state.turns, state.chat_revision)
            if revision is None:
                new_id = uuid4().hex
                revision = store.save(owner, new_id, state.turns, 0)
                state.chat_id = new_id
                state.history_notice = "다른 탭에서 대화가 변경되어 이 대화를 별도 기록으로 저장했어요."
            state.chat_revision = revision
            state.session_chats.pop(original_id, None)
            return
        except conversation_history.DeletedConversation:
            state.session_chats.pop(original_id, None)
            reset_chat()
            state.history_notice = "이 대화는 다른 탭에서 삭제됐어요."
            return
        except sqlite3.Error:
            state.history_notice = "대화를 저장할 수 없어 현재 접속에 기록을 보관했어요."
    state.session_chats[state.chat_id] = (state.turns, state.chat_revision)


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
    state.upload_signature = None
    state.attachment = None
    state.attachment_error = None


def reuse_document(attachment):
    clear_attachment()
    state.attachment = attachment


def reset_chat():
    state.chat_id = uuid4().hex
    state.chat_revision = 0
    state.turns = []
    state.pop("answer_action", None)
    state.pop("retry", None)
    clear_attachment()


def new_chat():
    if state.turns and not state.chat_revision:
        persist_chat()
    reset_chat()
    state.question = ""


if store and state.chat_id in state.session_chats:
    persist_chat()


def open_chat(chat_id):
    if chat_id == state.chat_id:
        return
    if chat_id in state.session_chats:
        loaded = state.session_chats[chat_id]
    else:
        try:
            loaded = store.load(owner, chat_id) if store else None
        except sqlite3.Error:
            loaded = None
    if not loaded:
        state.history_notice = "이 대화를 열 수 없어요. 기록 목록에서 다른 대화를 선택해 주세요."
        return
    new_chat()
    state.chat_id = chat_id
    state.turns, state.chat_revision = loaded


def delete_chat():
    if not store and state.chat_revision:
        state.history_notice = "저장소에 연결할 수 없어 대화를 삭제하지 못했어요."
        return
    if store:
        try:
            store.delete(owner, state.chat_id)
        except sqlite3.Error:
            state.history_notice = "대화를 삭제할 수 없어요. 잠시 후 다시 시도해 주세요."
            return
    state.session_chats.pop(state.chat_id, None)
    state.turns = []
    new_chat()


def render_history():
    with st.popover("대화 기록", icon=":material/menu:"):
        if store:
            st.caption("이 브라우저에서 대화를 다시 열 수 있어요.")
        else:
            st.caption("현재 접속에서 대화를 다시 열 수 있어요.")
        chats = {
            key: conversation_history.chat_title(turns)
            for key, (turns, _) in reversed(state.session_chats.items())
        }
        if store:
            try:
                chats.update(store.list_chats(owner))
            except sqlite3.Error:
                state.history_notice = "저장한 대화를 불러올 수 없어요."
        if state.turns and state.chat_id not in chats:
            chats[state.chat_id] = conversation_history.chat_title(state.turns)
        if state.history_notice:
            st.caption(state.history_notice)
        if not chats:
            st.caption("질문을 보내면 대화가 여기에 표시돼요.")
            return
        with st.container(height=220, border=False):
            for chat_id, title in chats.items():
                st.button(title, key=f"history_{chat_id}", on_click=open_chat, args=(chat_id,),
                          type="secondary" if chat_id == state.chat_id else "tertiary", width="stretch")
        if state.turns:
            with st.expander("현재 대화 관리"):
                st.download_button(
                    "전체 대화 저장", conversation_history.export_chat(state.turns),
                    file_name="ai-studio-conversation.md", mime="text/markdown", on_click="ignore",
                )
                st.button("현재 대화 삭제", on_click=delete_chat)


def set_question(text):
    state.question = text


def attachment_caption(attachment):
    if attachment.image:
        width, height = attachment.dimensions
        return f"이미지 준비 완료 · {width:,} × {height:,} px"
    if attachment.total_pages is not None:
        return f"{attachment.total_pages:,}쪽 중 {len(attachment.sources):,}쪽의 텍스트를 읽었어요."
    if attachment.truncated:
        return f"문서에서 텍스트 {sum(len(source.text) for source in attachment.sources):,}자를 읽었어요. 일부만 답변에 사용됩니다."
    return "문서에서 텍스트를 읽었습니다."


def coverage_note(attachment):
    if attachment.total_pages is None:
        return "이 답변에는 문서의 일부만 사용되었습니다." if attachment.truncated else ""
    skipped = attachment.unreadable_pages
    notes = []
    if skipped:
        pages = ", ".join(str(page) for page in skipped[:8])
        remaining = f" 외 {len(skipped) - 8:,}쪽" if len(skipped) > 8 else ""
        notes.append(f"텍스트를 읽지 못한 쪽 · {pages}{remaining}")
    if attachment.limit_reached:
        remaining = attachment.total_pages - len(attachment.sources) - len(skipped)
        if attachment.partial_page:
            notes.append(f"{attachment.partial_page:,}쪽은 분석 한도로 일부만 사용됐어요.")
        if remaining:
            notes.append(f"분석 한도로 확인하지 못한 쪽 · {remaining:,}쪽")
    return " / ".join(notes)


def render_attachment():
    attachment = state.attachment
    if not attachment and not state.attachment_error:
        return

    with st.container(key="attachment-card", horizontal=True, vertical_alignment="center"):
        with st.container():
            st.text(attachment.name if attachment else state.upload_signature[0])
            if state.attachment_error:
                st.error(state.attachment_error)
            else:
                st.caption(attachment_caption(attachment))
                if attachment.total_pages is not None and (note := coverage_note(attachment)):
                    st.caption(note)

        if attachment:
            with st.popover("미리보기", type="tertiary"):
                if attachment.image:
                    st.image(attachment.image, width=280)
                else:
                    st.caption(attachment.sources[0].label)
                    st.text(attachment.sources[0].text[:800])
        st.button("제거", type="tertiary", on_click=clear_attachment)
    if attachment:
        data = "이미지" if attachment.image else "문서에서 읽은 텍스트"
        st.caption(f"질문을 보내면 {data}가 {state.provider}에 전달됩니다.")


def request_error(exc, provider):
    if isinstance(exc, (APIConnectionError, TransportError)):
        return f"{provider}에 연결할 수 없습니다. 잠시 후 다시 시도해 주세요."
    if isinstance(exc, APIStatusError):
        if exc.status_code in (401, 403):
            return f"설정에서 {provider} API 키와 모델 접근 권한을 확인해 주세요."
        if exc.status_code == 429:
            return f"{provider} 요청 한도에 도달했습니다. 잠시 후 다시 시도해 주세요."
        if exc.status_code in (400, 404, 413):
            return f"{provider}에서 입력을 처리할 수 없습니다. 첨부 파일과 모델 설정을 확인해 주세요."
    return "답변을 가져오는 중 문제가 생겼습니다. 잠시 후 다시 시도해 주세요."


def generate_answer(turn, history, *, continuing=False):
    provider = state.provider
    config = PROVIDERS[provider]
    key = state.api_keys.get(provider) or server_key(provider)
    existing = turn.answer if continuing else ""
    turn.answer = existing
    turn.error = None
    turn.status = "interrupted"
    if not key:
        turn.error = f"설정에서 {provider} API 키를 입력한 뒤 다시 시도해 주세요."
        turn.status = "error"
        persist_chat()
        return

    text_model, vision_model = state.provider_settings.get(provider) or model_defaults(provider)
    model, messages = build_request(
        turn, history, text_model, vision_model,
        language=LANGUAGES[state.answer_language], length=LENGTH_INSTRUCTIONS.get(state.answer_length, ""),
        continuation=existing,
    )
    parts = []
    finish_reason = None

    def read_tokens(stream):
        nonlocal finish_reason
        for chunk in stream:
            if chunk.choices:
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                if choice.delta.content:
                    yield choice.delta.content

    try:
        with OpenAI(base_url=config.base_url, api_key=key, timeout=35, max_retries=0) as client:
            with st.spinner("답변을 준비하고 있어요"):
                stream = client.chat.completions.create(
                    model=model, messages=messages, stream=True,
                    **{config.token_parameter: TOKEN_LIMITS[state.answer_length]},
                )
                tokens = read_tokens(stream)
                first_token = next(tokens, None)

            if first_token:
                def output():
                    for text in chain((first_token,), tokens):
                        parts.append(text)
                        yield text

                st.write_stream(output())
            if finish_reason == "content_filter":
                turn.status = "filtered"
            elif not parts:
                turn.error = f"{provider}에서 표시할 답변을 받지 못했어요. 다시 시도해 주세요."
                turn.status = "error"
            elif finish_reason == "length":
                turn.status = "length"
            elif finish_reason == "stop":
                turn.status = "complete"
            else:
                turn.status = "interrupted"
    except (OpenAIError, TransportError) as exc:
        LOG.warning("%s request failed (%s, status %s)", provider, type(exc).__name__, getattr(exc, "status_code", None))
        turn.error = request_error(exc, provider)
        turn.status = "error"
    finally:
        turn.answer = existing + "".join(parts)
        persist_chat()


def render_turn(turn, index, action=None):
    with st.container(key=f"user-turn-{index}"), st.chat_message("user", avatar=":material/person:"):
        if turn.attachment:
            st.text(f"첨부 · {turn.attachment.name}")
        st.markdown(turn.question)

    with st.container(key=f"assistant-turn-{index}"), st.chat_message("assistant", avatar=":material/auto_awesome:"):
        if action:
            if action == "continue" and turn.answer:
                st.markdown(turn.answer)
            generate_answer(turn, state.turns[:index], continuing=action == "continue")
            return

        attachment = turn.attachment
        references = cited_sources(attachment, turn.answer) if attachment and attachment.sources else []
        if turn.answer:
            st.markdown(citation_links(turn.answer, references, index) if references else turn.answer)
        if turn.error:
            st.error(turn.error)
        elif turn.status in INCOMPLETE_LABELS:
            st.caption(INCOMPLETE_LABELS[turn.status])

        if turn.status in {"error", "length", "interrupted"} or turn.error:
            if attachment and attachment.mime_type and not attachment.image:
                st.caption("이미지를 다시 첨부하면 새 질문을 보낼 수 있어요.")
            else:
                with st.container(horizontal=True):
                    if turn.answer:
                        st.button("답변 이어쓰기", key=f"continue_{index}", on_click=state.update,
                                  args=({"answer_action": (index, "continue")},))
                    st.button("다시 시도", key=f"retry_{index}", on_click=state.update,
                              args=({"answer_action": (index, "retry")},))
        if not turn.answer:
            return

        if attachment and attachment.sources and (note := coverage_note(attachment)):
            st.caption(note)
        if references:
            with st.expander(f"문서 근거 {len(references)}개 보기"):
                for number, source in references:
                    st.html(f'<div id="studio-source-{index}-{number}" tabindex="-1" aria-label="문서 근거 {number}"></div>')
                    st.text(f"[{number}] {source.label} · {attachment.name}")
                    st.text(source.text)

        with st.container(horizontal=True, key=f"answer-actions-{index}"):
            with st.popover("복사", type="tertiary", icon=":material/content_copy:"):
                st.code(turn.answer, language=None, wrap_lines=True)
            st.download_button(
                "저장", turn.answer, file_name="ai-studio-answer.md", mime="text/markdown",
                key=f"download_{index}", on_click="ignore", type="tertiary", icon=":material/download:",
            )
        if turn.status == "complete" and index == len(state.turns) - 1 and attachment and attachment is state.attachment:
            with st.container(horizontal=True, key=f"followups-{index}"):
                if attachment.image:
                    st.button("이미지 속 글자 추출", key=f"followup_{index}", on_click=set_question,
                              args=("이 이미지에서 읽을 수 있는 글자를 추출해 줘.",))
                else:
                    st.button("근거를 더 자세히 설명", key=f"followup_{index}", on_click=set_question,
                              args=("방금 답변에서 사용한 문서 근거를 더 자세히 설명해 줘.",))
                    st.button("표로 다시 정리", key=f"followup_table_{index}", on_click=set_question,
                              args=("방금 답변의 핵심 내용을 표로 다시 정리해 줘.",))
        elif index == len(state.turns) - 1 and attachment and attachment.sources and not state.attachment and not state.attachment_error:
            st.button("이 문서로 계속 질문", key=f"reuse_document_{index}", on_click=reuse_document, args=(attachment,))


st.html(f"<style>{(ROOT / 'styles.css').read_text(encoding='utf-8')}</style>")
templates = Environment(loader=FileSystemLoader(ROOT / "templates"), autoescape=select_autoescape())
with st.container(key="studio-header", horizontal=True, vertical_alignment="center"):
    render_history()
    st.html(templates.get_template("header.html").render(title="AI Studio"))
    with st.container(key="studio-navigation", horizontal=True, vertical_alignment="center", width="content"):
        st.button("새 대화", icon=":material/add:", on_click=new_chat)
        render_settings()

transcript = st.container(key="studio-transcript")
welcome = transcript.empty()

with st.bottom:
    with st.container(key="studio-composer"):
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
            disabled=bool(state.attachment_error), submit_mode="stop",
        )
        if state.api_keys.get(state.provider) or server_key(state.provider):
            st.caption(f"{state.provider} 사용 중")
        else:
            st.caption(f"설정에서 {state.provider} API 키를 입력하세요.")

if not state.turns:
    with welcome.container(), st.container(key="studio-welcome"):
        st.html('<h2 class="studio-welcome-title">문서나 이미지에 대해<br>무엇이 궁금한가요?</h2>')
        st.caption("바로 질문하거나 파일을 첨부해 보세요.")
        if state.show_suggestions:
            labels = SUGGESTIONS.keys()
            if state.attachment:
                labels = ("이미지 설명",) if state.attachment.image else ("문서 요약", "문서에서 항목 찾기")
            with st.container(horizontal=True, key="studio-suggestions"):
                for label in labels:
                    st.button(label, on_click=set_question, args=(SUGGESTIONS[label],))

action_index, action = state.pop("answer_action", (None, None))
with transcript:
    for index, turn in enumerate(state.turns):
        render_turn(turn, index, action=action if index == action_index else None)
if action:
    st.rerun()

if question and question.strip() and not state.attachment_error:
    turn = Turn(question.strip(), state.attachment)
    state.turns.append(turn)
    welcome.empty()
    with transcript:
        render_turn(turn, len(state.turns) - 1, action="generate")
    st.rerun()
