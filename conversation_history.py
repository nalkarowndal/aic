import json
import os
import re
import sqlite3
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

import streamlit as st

from chat_service import Attachment, Source, Turn


_browser_bridge = st.components.v2.component(
    "ai_studio.browser_bridge",
    html='<span hidden></span>',
    js="""
    export default function ({ parentElement, setStateValue, setTriggerValue }) {
        const element = parentElement.querySelector('span');
        if (!element.dataset.sent) {
            element.dataset.sent = 'true';
            const storageKey = 'ai-studio-history-owner-v1';
            try {
                let value = localStorage.getItem(storageKey);
                if (!/^[0-9a-f]{64}$/.test(value || '')) {
                    value = Array.from(crypto.getRandomValues(new Uint8Array(32)),
                        byte => byte.toString(16).padStart(2, '0')).join('');
                    localStorage.setItem(storageKey, value);
                }
                setStateValue('owner', value);
            } catch {
                setStateValue('owner', 'unavailable');
            }
        }
        let stopObserver;
        const stopSelector = '[data-testid="stChatInputStopButton"]';
        function handleChatClick(event) {
            const stopButton = event.target.closest(stopSelector);
            if (stopButton) {
                stopObserver?.disconnect();
                // Streamlit can rerun only after the stopped script has finished.
                stopObserver = new MutationObserver(() => {
                    const input = document.querySelector('.st-key-studio-composer textarea');
                    if (!input || input.disabled || document.querySelector(stopSelector)) return;
                    stopObserver.disconnect();
                    setTriggerValue('stopped', true);
                });
                stopObserver.observe(document.body, {
                    childList: true, subtree: true, attributes: true, attributeFilter: ['data-testid', 'disabled'],
                });
                return;
            }

            const link = event.target.closest('.st-key-studio-transcript a[href^="#studio-source-"]');
            if (!link) return;
            const target = document.getElementById(link.hash.slice(1));
            if (!target) return;
            const details = target.closest('details');
            event.preventDefault();
            if (details && !details.open) details.querySelector('summary')?.click();
            requestAnimationFrame(() => {
                target.focus({ preventScroll: true });
                target.scrollIntoView({ block: 'nearest' });
            });
        }
        document.addEventListener('click', handleChatClick, true);
        return () => {
            stopObserver?.disconnect();
            document.removeEventListener('click', handleChatClick, true);
        };
    }
    """,
)


def browser_owner():
    with st.container(key="history-identity"):
        result = _browser_bridge(key="history_browser", on_owner_change=lambda: None, on_stopped_change=lambda: None)
    owner = result.get("owner")
    return owner if isinstance(owner, str) and re.fullmatch(r"[0-9a-f]{64}", owner) else None


def history_path():
    value = os.environ.get("AI_STUDIO_HISTORY_DB")
    return Path(value) if value else Path(__file__).resolve().parent / ".aic-data" / "history.sqlite3"


def chat_title(turns):
    text = " ".join(turns[0].question.split())
    return f"{text[:41]}…" if len(text) > 42 else text


def export_chat(turns):
    sections = [f"# {chat_title(turns)}"]
    for turn in turns:
        attachment = f"\n\n첨부 · {turn.attachment.name}" if turn.attachment else ""
        sections.append(f"## 질문\n\n{turn.question}{attachment}")
        if turn.answer:
            sections.append(f"## AI Studio\n\n{turn.answer}")
    return "\n\n".join(sections)


def serialize(turns):
    attachments = []
    indices = {}
    messages = []
    for turn in turns:
        attachment_id = None
        if turn.attachment:
            identity = id(turn.attachment)
            if identity not in indices:
                attachment = asdict(turn.attachment)
                del attachment["image"]
                indices[identity] = len(attachments)
                attachments.append(attachment)
            attachment_id = indices[identity]
        messages.append({
            "question": turn.question, "answer": turn.answer, "error": turn.error,
            "status": turn.status, "attachment": attachment_id,
        })
    return json.dumps({"attachments": attachments, "turns": messages}, ensure_ascii=False)


def deserialize(payload):
    data = json.loads(payload)
    attachments = []
    for item in data["attachments"]:
        item["sources"] = tuple(Source(**source) for source in item["sources"])
        item["unreadable_pages"] = tuple(item.get("unreadable_pages", ()))
        if item.get("dimensions"):
            item["dimensions"] = tuple(item["dimensions"])
        attachments.append(Attachment(**item))
    turns = []
    for item in data["turns"]:
        index = item.pop("attachment")
        turns.append(Turn(**item, attachment=attachments[index] if index is not None else None))
    return turns


class DeletedConversation(Exception):
    pass


class HistoryStore:
    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as connection, connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    owner TEXT NOT NULL,
                    id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    updated TEXT NOT NULL,
                    PRIMARY KEY (owner, id)
                )
            """)

    def connect(self):
        return sqlite3.connect(self.path, timeout=2)

    def save(self, owner, chat_id, turns, revision):
        title, payload = chat_title(turns), serialize(turns)
        with closing(self.connect()) as connection, connection:
            if revision == 0:
                cursor = connection.execute("""
                    INSERT OR IGNORE INTO conversations (owner, id, title, payload, revision, updated)
                    VALUES (?, ?, ?, ?, 1, strftime('%Y-%m-%dT%H:%M:%f', 'now'))
                """, (owner, chat_id, title, payload))
            else:
                cursor = connection.execute("""
                    UPDATE conversations SET title = ?, payload = ?, revision = revision + 1,
                        updated = strftime('%Y-%m-%dT%H:%M:%f', 'now')
                    WHERE owner = ? AND id = ? AND revision = ?
                """, (title, payload, owner, chat_id, revision))
            if cursor.rowcount:
                return revision + 1
            if revision and not connection.execute(
                "SELECT 1 FROM conversations WHERE owner = ? AND id = ?", (owner, chat_id),
            ).fetchone():
                raise DeletedConversation
            return None

    def list_chats(self, owner):
        with closing(self.connect()) as connection:
            return connection.execute("""
                SELECT id, title FROM conversations WHERE owner = ? ORDER BY updated DESC, id
            """, (owner,)).fetchall()

    def load(self, owner, chat_id):
        with closing(self.connect()) as connection:
            row = connection.execute("""
                SELECT payload, revision FROM conversations WHERE owner = ? AND id = ?
            """, (owner, chat_id)).fetchone()
        return (deserialize(row[0]), row[1]) if row else None

    def delete(self, owner, chat_id):
        with closing(self.connect()) as connection, connection:
            connection.execute("DELETE FROM conversations WHERE owner = ? AND id = ?", (owner, chat_id))
