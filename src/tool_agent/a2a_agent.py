"""A2A wrapper for ToolAgent."""

from __future__ import annotations

from typing import Any

from Bonus.a2a import A2AMessage, BaseAgent
from .tool_agent import ToolAgent


class ToolA2AAgent(BaseAgent):
    agent_name = "tool_agent"

    def __init__(self, tool_agent: ToolAgent) -> None:
        self.tool_agent = tool_agent

    async def process_message(self, message: A2AMessage, *, ui: Any | None = None) -> A2AMessage:
        routed_task = message.payload.get("routed_task") or {}
        user_input = str(message.payload.get("user_input") or routed_task.get("task_description") or "")
        result = await self.tool_agent.run(routed_task, ui, user_input=user_input)
        return message.create_reply(
            sender=self.agent_name,
            payload={"status": result.get("status", "handled"), "result": result},
            message_type="tool.task.result",
        )
