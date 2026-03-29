"""A2A Agent 注册表与派发器。"""

from __future__ import annotations

from typing import Any

from .base import BaseAgent
from .message import A2AMessage


class AgentRegistry:
    """中心化 Agent 注册与动态路由。"""

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        self._agents[agent.agent_name] = agent

    def get(self, name: str) -> BaseAgent | None:
        return self._agents.get(name)

    def registered_names(self) -> list[str]:
        return sorted(self._agents)

    async def dispatch(self, message: A2AMessage, *, ui: Any | None = None) -> A2AMessage:
        agent = self.get(message.receiver)
        if agent is None:
            return message.create_reply(
                sender="agent_registry",
                ok=False,
                error=f"未注册的目标 Agent: {message.receiver}",
                payload={"available_agents": self.registered_names()},
                message_type="a2a.error",
            )
        try:
            return await agent.process_message(message, ui=ui)
        except Exception as exc:
            return message.create_reply(
                sender="agent_registry",
                ok=False,
                error=f"{message.receiver} 执行失败: {exc}",
                payload={"status": "dispatch_error"},
                message_type="a2a.error",
            )
