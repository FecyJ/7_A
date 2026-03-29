"""A2A 版 Memory Agent。"""

from __future__ import annotations

from typing import Any

from ..a2a import A2AMessage, BaseAgent
from .service import MemoryService


class MemoryAgent(BaseAgent):
    agent_name = "memory_agent"

    def __init__(self, service: MemoryService | None = None) -> None:
        self.service = service or MemoryService()

    async def process_message(self, message: A2AMessage, *, ui: Any | None = None) -> A2AMessage:
        del ui
        available, reason = self.service.is_available()
        if not available:
            return message.create_reply(
                sender=self.agent_name,
                ok=False,
                error=f"Memory Agent 不可用: {reason}",
                payload={"status": "unavailable"},
                message_type="memory.error",
            )

        action = str(message.payload.get("action") or "").strip().lower()
        content = str(message.payload.get("content") or "").strip()
        category = message.payload.get("category")
        top_k = int(message.payload.get("top_k") or 5)

        if action == "save":
            result = self.service.save(content, category if isinstance(category, str) else None)
        elif action == "search":
            result = self.service.search(content, top_k=top_k)
        elif action == "delete":
            result = self.service.delete(content)
        elif action == "list":
            result = self.service.list_all(category if isinstance(category, str) else None)
        else:
            return message.create_reply(
                sender=self.agent_name,
                ok=False,
                error=f"不支持的 memory action: {action}",
                payload={"status": "invalid_action"},
                message_type="memory.error",
            )

        return message.create_reply(
            sender=self.agent_name,
            payload={"status": "handled", **result},
            message_type=f"memory.{action}.result",
        )
