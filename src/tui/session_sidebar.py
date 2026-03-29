"""会话历史侧边栏组件。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from rich.text import Text
from textual.widgets import ListItem, ListView, Static


class SessionListItem(ListItem):
    """单个会话条目。"""

    DEFAULT_CSS = """
    SessionListItem {
        padding: 0 1;
        border-left: tall transparent;
        min-height: 3;
    }

    SessionListItem.-active {
        border-left: tall $success;
    }
    """

    def __init__(self, session_id: str, title: str, updated_at: str | None = None) -> None:
        self._body = Static("", classes="session-item-body")
        super().__init__(self._body)
        self.session_id = session_id
        self.title = title
        self.updated_at = updated_at or ""
        self._refresh_body()

    @staticmethod
    def _format_updated_at(value: str) -> str:
        if not value:
            return ""
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return value
        return dt.strftime("%m-%d %H:%M")

    def _refresh_body(self) -> None:
        text = Text()
        text.append(self.title or "新对话", style="bold")
        subtitle = self._format_updated_at(self.updated_at)
        if subtitle:
            text.append("\n")
            text.append(subtitle, style="color(245)")
        self._body.update(text)

    def update_session(self, title: str, updated_at: str | None = None) -> None:
        self.title = title
        self.updated_at = updated_at or self.updated_at
        self._refresh_body()

    def set_active(self, active: bool) -> None:
        self.set_class(active, "-active")


class SessionSidebar(ListView):
    """左侧会话列表。"""

    DEFAULT_CSS = """
    SessionSidebar {
        height: 1fr;
        border: round $panel-lighten-1;
        background: $panel;
    }
    """

    async def rebuild(
        self,
        sessions: list[dict[str, Any]],
        *,
        active_session_id: str | None = None,
    ) -> None:
        await self.remove_children()
        items: list[SessionListItem] = []
        active_index = 0 if sessions else None
        for index, session in enumerate(sessions):
            session_id = str(session.get("id") or "")
            item = SessionListItem(
                session_id=session_id,
                title=str(session.get("title") or "新对话"),
                updated_at=str(session.get("updated_at") or session.get("created_at") or ""),
            )
            item.set_active(session_id == active_session_id)
            items.append(item)
            if session_id == active_session_id:
                active_index = index

        if items:
            await self.extend(items)
            self.index = active_index
        else:
            self.index = None

    def set_active_session(self, session_id: str | None) -> None:
        target_index: int | None = None
        for index, child in enumerate(self.children):
            if not isinstance(child, SessionListItem):
                continue
            is_active = child.session_id == session_id
            child.set_active(is_active)
            if is_active:
                target_index = index
        self.index = target_index

    def session_id_at(self, index: int | None) -> str | None:
        if index is None or index < 0 or index >= len(self.children):
            return None
        item = self.children[index]
        if isinstance(item, SessionListItem):
            return item.session_id
        return None
