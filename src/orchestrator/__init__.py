"""Orchestrator package exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LLM_EXPORTS = {
    "API_MODE_ENV",
    "DEFAULT_MODEL",
    "generate_text_response",
    "get_api_mode",
    "get_default_model",
    "get_llm_client",
    "stream_text_response",
}

_ORCHESTRATOR_EXPORTS = {
    "OrchestratorAgent",
    "OrchestratorOutput",
    "build_controller",
    "execute_command_async",
    "main_controller",
}

__all__ = sorted(_LLM_EXPORTS | _ORCHESTRATOR_EXPORTS)


def __getattr__(name: str) -> Any:
    """Lazy-load exports to avoid circular imports."""
    if name in _LLM_EXPORTS:
        module = import_module(".llm_client", __name__)
        return getattr(module, name)
    if name in _ORCHESTRATOR_EXPORTS:
        module = import_module(".orchestrator", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
