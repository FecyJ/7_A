"""A2A wrapper for ShellAgent."""

from __future__ import annotations

from typing import Any

from Bonus.a2a import A2AMessage, BaseAgent
from .shell_agent import ShellAgent


class ShellA2AAgent(BaseAgent):
    agent_name = "shell_agent"

    def __init__(self, shell_agent: ShellAgent, shell_executor=None) -> None:
        self.shell_agent = shell_agent
        self.shell_executor = shell_executor

    async def process_message(self, message: A2AMessage, *, ui: Any | None = None) -> A2AMessage:
        mode = str(message.payload.get("mode") or "task")
        if mode == "direct_command":
            command = str(message.payload.get("command") or "").strip()
            result = await self.shell_agent.run_direct_command(command, ui, self.shell_executor)
            return message.create_reply(
                sender=self.agent_name,
                payload={"status": result.get("status", "handled"), "result": result},
                message_type="shell.direct_command.result",
            )

        routed_task = message.payload.get("routed_task") or {}
        user_input = str(message.payload.get("user_input") or routed_task.get("task_description") or "")
        result = await self.shell_agent.run(routed_task, ui, self.shell_executor, user_input=user_input)
        return message.create_reply(
            sender=self.agent_name,
            payload={"status": result.get("status", "handled"), "result": result},
            message_type="shell.task.result",
        )
