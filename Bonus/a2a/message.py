"""A2A 消息信封定义。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class A2AMessage(BaseModel):
    """系统内 Agent 间统一传递的消息信封。"""

    id: str = Field(default_factory=lambda: uuid4().hex)
    sender: str
    receiver: str
    message_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    in_reply_to: str | None = None
    ok: bool = True
    error: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def create_reply(
        self,
        *,
        sender: str,
        payload: dict[str, Any] | None = None,
        message_type: str | None = None,
        ok: bool = True,
        error: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> "A2AMessage":
        """基于当前消息构造响应消息。"""
        return A2AMessage(
            sender=sender,
            receiver=self.sender,
            message_type=message_type or f"{self.message_type}.result",
            payload=payload or {},
            context=context or {},
            in_reply_to=self.id,
            ok=ok,
            error=error,
        )
