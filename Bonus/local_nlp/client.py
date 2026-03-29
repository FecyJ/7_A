"""本地 NLP Bonus：双引擎客户端初始化。"""

from __future__ import annotations

import os

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI


load_dotenv()

LOCAL_SLM_BASE_URL = os.getenv("LOCAL_SLM_BASE_URL", "http://localhost:11434/v1")
LOCAL_SLM_MODEL = os.getenv("LOCAL_SLM_MODEL", "qwen3.5:4b")
LOCAL_SLM_API_KEY = os.getenv("LOCAL_SLM_API_KEY", "ollama")
LOCAL_SLM_TIMEOUT_SECONDS = float(os.getenv("LOCAL_SLM_TIMEOUT_SECONDS", "8"))

CLOUD_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
_CLOUD_API_KEY = os.getenv("OPENAI_API_KEY", "placeholder")
_CLOUD_BASE_URL = os.getenv("OPENAI_BASE_URL")

cloud_async_client = AsyncOpenAI(
    api_key=_CLOUD_API_KEY,
    base_url=_CLOUD_BASE_URL,
)

local_slm_client = AsyncOpenAI(
    base_url=LOCAL_SLM_BASE_URL,
    api_key=LOCAL_SLM_API_KEY,
    timeout=httpx.Timeout(LOCAL_SLM_TIMEOUT_SECONDS),
)


__all__ = [
    "CLOUD_MODEL",
    "LOCAL_SLM_BASE_URL",
    "LOCAL_SLM_MODEL",
    "LOCAL_SLM_TIMEOUT_SECONDS",
    "cloud_async_client",
    "local_slm_client",
]
