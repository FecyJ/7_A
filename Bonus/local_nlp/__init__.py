"""Bonus/local_nlp exports."""

from .client import (
    CLOUD_MODEL,
    LOCAL_SLM_BASE_URL,
    LOCAL_SLM_MODEL,
    LOCAL_SLM_TIMEOUT_SECONDS,
    cloud_async_client,
    local_slm_client,
)
from .router import (
    DualEngineRouter,
    LOCAL_ALLOW_DIRECT_ANSWER_FASTPATH,
    LOCAL_FASTPATH_JSON_SCHEMA,
    LOCAL_FASTPATH_JSON_SCHEMA_NAME,
    LOCAL_PASS_CONFIDENCE,
    get_local_context,
    get_local_system_prompt,
    normalize_local_result,
    route_user_input,
    try_local_fastpath,
)

__all__ = [
    "CLOUD_MODEL",
    "DualEngineRouter",
    "LOCAL_ALLOW_DIRECT_ANSWER_FASTPATH",
    "LOCAL_FASTPATH_JSON_SCHEMA",
    "LOCAL_FASTPATH_JSON_SCHEMA_NAME",
    "LOCAL_PASS_CONFIDENCE",
    "LOCAL_SLM_BASE_URL",
    "LOCAL_SLM_MODEL",
    "LOCAL_SLM_TIMEOUT_SECONDS",
    "cloud_async_client",
    "get_local_context",
    "get_local_system_prompt",
    "local_slm_client",
    "normalize_local_result",
    "route_user_input",
    "try_local_fastpath",
]
