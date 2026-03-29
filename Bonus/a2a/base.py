"""A2A Agent 基类契约。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .message import A2AMessage


class BaseAgent(ABC):
    """所有可挂到消息总线上的 Agent 都应实现该接口。"""

    agent_name: str

    @abstractmethod
    async def process_message(self, message: A2AMessage, *, ui: Any | None = None) -> A2AMessage:
        """处理一条请求消息并返回响应消息。"""
        raise NotImplementedError
