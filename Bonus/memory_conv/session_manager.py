"""Session History 持久化与标题生成。"""

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from Bonus.local_nlp.client import LOCAL_SLM_MODEL, local_slm_client
except Exception:  # pragma: no cover - 本地模型环境可选
    LOCAL_SLM_MODEL = "qwen3.5:4b"  # type: ignore[assignment]
    local_slm_client = None  # type: ignore[assignment]


class SessionManager:
    """管理会话列表、消息持久化与本地标题生成。"""

    DEFAULT_TITLE = "新对话"

    def __init__(self, storage_path: str | Path | None = None) -> None:
        base_dir = Path(__file__).resolve().parent
        self.storage_path = Path(storage_path or (base_dir / "sessions.json"))

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat()

    def _empty_store(self) -> dict[str, Any]:
        return {
            "active_session_id": None,
            "sessions": [],
        }

    def _ensure_storage(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            self.storage_path.write_text(
                json.dumps(self._empty_store(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _load_store(self) -> dict[str, Any]:
        self._ensure_storage()
        try:
            raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except Exception:
            raw = self._empty_store()
        if not isinstance(raw, dict):
            raw = self._empty_store()
        sessions = raw.get("sessions")
        if not isinstance(sessions, list):
            raw["sessions"] = []
        raw.setdefault("active_session_id", None)
        return raw

    def _save_store(self, data: dict[str, Any]) -> None:
        self._ensure_storage()
        self.storage_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _normalize_messages(messages: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in list(messages or []):
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "").strip()
            content = str(item.get("content") or "")
            if not kind:
                continue
            normalized_item = {
                "kind": kind,
                "content": content,
            }
            for key in ("style", "state", "language"):
                value = item.get(key)
                if value is not None:
                    normalized_item[key] = str(value)
            for key in ("markdown",):
                value = item.get(key)
                if value is not None:
                    normalized_item[key] = bool(value)
            normalized.append(normalized_item)
        return normalized

    @staticmethod
    def _normalize_orchestrator_state(state: dict[str, Any] | None) -> dict[str, Any]:
        state = dict(state or {})
        normalized = {
            "chat_history": [],
            "short_term_memory": [],
            "last_file_path": state.get("last_file_path"),
            "last_document_content": state.get("last_document_content"),
            "last_document_summary": state.get("last_document_summary"),
        }

        for key in ("chat_history", "short_term_memory"):
            items: list[dict[str, Any]] = []
            for item in list(state.get(key) or []):
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role") or "").strip()
                content = str(item.get("content") or "")
                if not role or not content:
                    continue
                items.append({"role": role, "content": content})
            normalized[key] = items

        return normalized

    def list_sessions(self) -> list[dict[str, Any]]:
        store = self._load_store()
        sessions = [session for session in store.get("sessions", []) if isinstance(session, dict)]
        sessions.sort(
            key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""),
            reverse=True,
        )
        return deepcopy(sessions)

    def get_active_session_id(self) -> str | None:
        store = self._load_store()
        active = store.get("active_session_id")
        return str(active) if active else None

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        for session in self.list_sessions():
            if str(session.get("id")) == str(session_id):
                return session
        return None

    def create_session(self, *, title: str | None = None, activate: bool = True) -> dict[str, Any]:
        store = self._load_store()
        now = self._now()
        session = {
            "id": uuid.uuid4().hex[:12],
            "title": str(title or self.DEFAULT_TITLE),
            "created_at": now,
            "updated_at": now,
            "messages": [],
            "orchestrator_state": self._normalize_orchestrator_state(None),
        }
        store.setdefault("sessions", []).append(session)
        if activate:
            store["active_session_id"] = session["id"]
        self._save_store(store)
        return deepcopy(session)

    def set_active_session(self, session_id: str) -> None:
        store = self._load_store()
        target = next(
            (session for session in store.get("sessions", []) if str(session.get("id")) == str(session_id)),
            None,
        )
        if target is None:
            return
        store["active_session_id"] = str(session_id)
        self._save_store(store)

    def save_session_snapshot(
        self,
        session_id: str,
        *,
        messages: list[dict[str, Any]] | None,
        orchestrator_state: dict[str, Any] | None,
        title: str | None = None,
    ) -> dict[str, Any] | None:
        store = self._load_store()
        sessions = store.setdefault("sessions", [])
        target = next(
            (session for session in sessions if isinstance(session, dict) and str(session.get("id")) == str(session_id)),
            None,
        )
        if target is None:
            return None

        target["messages"] = self._normalize_messages(messages)
        target["orchestrator_state"] = self._normalize_orchestrator_state(orchestrator_state)
        target["updated_at"] = self._now()
        if title is not None:
            target["title"] = str(title).strip() or self.DEFAULT_TITLE
        else:
            target["title"] = str(target.get("title") or self.DEFAULT_TITLE)

        store["active_session_id"] = str(session_id)
        self._save_store(store)
        return deepcopy(target)

    def update_session_title(self, session_id: str, title: str) -> dict[str, Any] | None:
        cleaned = self._sanitize_title(title) or self.DEFAULT_TITLE
        store = self._load_store()
        target = next(
            (
                session
                for session in store.get("sessions", [])
                if isinstance(session, dict) and str(session.get("id")) == str(session_id)
            ),
            None,
        )
        if target is None:
            return None
        target["title"] = cleaned
        target["updated_at"] = self._now()
        self._save_store(store)
        return deepcopy(target)

    @staticmethod
    def _fallback_title_from_history(chat_history: list[dict[str, Any]]) -> str:
        for item in chat_history:
            if str(item.get("role") or "") != "user":
                continue
            content = " ".join(str(item.get("content") or "").split())
            if content:
                return content[:14].rstrip("，。,:：?？!！") or SessionManager.DEFAULT_TITLE
        return SessionManager.DEFAULT_TITLE

    @staticmethod
    def _sanitize_title(title: str) -> str:
        cleaned = " ".join(str(title or "").replace("\n", " ").split())
        cleaned = cleaned.strip("「」『』【】[]()（）\"'` ")
        if not cleaned:
            return ""
        if len(cleaned) > 18:
            cleaned = cleaned[:18].rstrip()
        return cleaned

    async def generate_title_for_session(
        self,
        session_id: str,
        *,
        chat_history: list[dict[str, Any]] | None = None,
    ) -> str | None:
        """异步生成简短标题，并写回 sessions.json。"""
        session = self.get_session(session_id)
        if session is None:
            return None

        current_title = str(session.get("title") or "").strip()
        if current_title and current_title != self.DEFAULT_TITLE:
            return current_title

        history = list(chat_history or session.get("orchestrator_state", {}).get("chat_history") or [])
        if not history:
            return None

        generated_title = ""
        if local_slm_client is not None:
            transcript_lines: list[str] = []
            for item in history[-8:]:
                if not isinstance(item, dict):
                    continue
                role = "用户" if str(item.get("role") or "") == "user" else "助手"
                content = " ".join(str(item.get("content") or "").split())
                if content:
                    transcript_lines.append(f"{role}: {content}")
            prompt = "\n".join(transcript_lines)

            try:
                response = await local_slm_client.chat.completions.create(
                    model=LOCAL_SLM_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "你是会话标题生成器。"
                                "请基于下面的对话内容生成一个简短标题。"
                                "要求：中文优先，长度不超过12个汉字或18个字符，只输出标题本身，不要加引号、序号或解释。"
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature=0.2,
                    max_tokens=24,
                )
                generated_title = (
                    response.choices[0].message.content
                    if response and response.choices
                    else ""
                ) or ""
            except Exception:
                generated_title = ""

        final_title = self._sanitize_title(generated_title) or self._fallback_title_from_history(history)
        updated = self.update_session_title(session_id, final_title)
        return str(updated.get("title")) if isinstance(updated, dict) else final_title
