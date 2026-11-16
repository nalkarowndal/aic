import os
from dataclasses import dataclass

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from chat_service import GROQ_URL, TEXT_MODEL, VISION_MODEL


@dataclass(frozen=True)
class Provider:
    base_url: str
    text_model: str
    vision_model: str
    token_parameter: str = "max_completion_tokens"


PROVIDERS = {
    "Groq": Provider(GROQ_URL, TEXT_MODEL, VISION_MODEL),
    "OpenAI": Provider("https://api.openai.com/v1", "gpt-4.1-mini", "gpt-4.1-mini"),
    "Gemini": Provider(
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "gemini-4-hydrogen", "gemini-4-hydrogen", token_parameter="max_tokens",
    ),
}
TOKEN_LIMITS = {"짧게": 1_500, "기본": 3_000, "자세히": 6_000}
LANGUAGES = {"질문에 맞춤": "", "한국어": "Korean", "English": "English"}


def server_setting(name):
    value = os.environ.get(name)
    if not value:
        try:
            value = st.secrets.get(name, "")
        except StreamlitSecretNotFoundError:
            return ""
    return str(value).strip()


def server_key(provider):
    return server_setting(f"{provider.upper()}_API_KEY")


def model_defaults(provider):
    prefix = provider.upper()
    config = PROVIDERS[provider]
    return (
        server_setting(f"{prefix}_TEXT_MODEL") or config.text_model,
        server_setting(f"{prefix}_VISION_MODEL") or config.vision_model,
    )


def init_settings(state):
    state.setdefault("provider", "Groq")
    state.setdefault("provider_settings", {})
    state.setdefault("api_keys", {})
    state.setdefault("answer_length", "기본")
    state.setdefault("answer_language", "질문에 맞춤")
    state.setdefault("show_suggestions", True)


def apply_settings(provider, fields):
    state = st.session_state
    key_field, text_field, vision_field, length_field, language_field, suggestions_field = fields
    if key := state[key_field].strip():
        state.api_keys[provider] = key
    state[key_field] = ""
    text_default, vision_default = model_defaults(provider)
    state.provider_settings[provider] = (
        state[text_field].strip() or text_default,
        state[vision_field].strip() or vision_default,
    )
    state.provider = provider
    state.answer_length = state[length_field]
    state.answer_language = state[language_field]
    state.show_suggestions = state[suggestions_field]


def remove_key(provider):
    st.session_state.api_keys.pop(provider, None)
    st.session_state[f"settings_{provider}_key"] = ""


def render_settings():
    state = st.session_state
    with st.popover("설정", icon=":material/settings:"):
        form_area = st.container()
        with st.expander("연결 변경"):
            provider = st.radio(
                "연결할 서비스", tuple(PROVIDERS), key="settings_provider",
                horizontal=True, label_visibility="collapsed",
            )
            if provider in state.api_keys:
                st.button("입력한 키 지우기", key=f"clear_key_{provider}", on_click=remove_key, args=(provider,))

        text_model, vision_model = state.provider_settings.get(provider) or model_defaults(provider)
        fields = tuple(f"settings_{provider}_{name}" for name in ("key", "text", "vision", "length", "language", "suggestions"))
        with form_area, st.form(f"settings_{provider}", enter_to_submit=False, border=False):
            st.text_input("API 키", type="password", key=fields[0], help="입력한 키는 현재 접속에서 사용합니다.")
            with st.expander("모델 및 답변 옵션"):
                st.text_input("텍스트 모델", value=text_model, key=fields[1])
                st.text_input("이미지 모델", value=vision_model, key=fields[2])
                st.selectbox("답변 길이", tuple(TOKEN_LIMITS), index=tuple(TOKEN_LIMITS).index(state.answer_length), key=fields[3])
                st.selectbox("답변 언어", tuple(LANGUAGES), index=tuple(LANGUAGES).index(state.answer_language), key=fields[4])
                st.checkbox("시작 질문 표시", value=state.show_suggestions, key=fields[5])
            st.form_submit_button("설정 적용", type="primary", on_click=apply_settings, args=(provider, fields))
