"""对 Bonus/memory 的结构化服务封装。"""

from __future__ import annotations

import importlib
import io
import logging
from contextlib import redirect_stderr, redirect_stdout
from typing import Any


class MemoryService:
    """将组员的 JSON+jieba 记忆引擎包装成可编排服务。"""

    def __init__(self) -> None:
        self._module = None
        self._import_error: Exception | None = None

    def _ensure_module(self):
        if self._module is not None:
            return self._module
        if self._import_error is not None:
            raise self._import_error
        try:
            self._module = importlib.import_module("Bonus.memory.memory_agent")
            try:
                jieba = importlib.import_module("jieba")
                if hasattr(jieba, "setLogLevel"):
                    jieba.setLogLevel(logging.ERROR)
            except Exception:
                pass
        except Exception as exc:  # pragma: no cover - 运行环境依赖项问题
            self._import_error = exc
            raise
        return self._module

    @staticmethod
    def _call_quietly(func, *args, **kwargs):
        sink = io.StringIO()
        with redirect_stdout(sink), redirect_stderr(sink):
            return func(*args, **kwargs)

    def is_available(self) -> tuple[bool, str | None]:
        try:
            self._ensure_module()
            return True, None
        except Exception as exc:
            return False, str(exc)

    def detect_intent(self, user_input: str) -> tuple[str | None, str]:
        module = self._ensure_module()
        action, content = module.detect_memory_intent(user_input)
        return action, content

    @staticmethod
    def _classify_category(content: str) -> str:
        text = (content or "").strip()
        if any(marker in text for marker in ["喜欢", "偏好", "习惯", "默认", "倾向", "常用"]):
            return "preference"
        if any(marker in text for marker in ["项目", "仓库", "框架", "技术栈", "依赖", "数据库"]):
            return "project"
        if any(marker in text for marker in ["是", "有", "叫", "地址", "端口", "密码", "账号", "IP"]):
            return "fact"
        return "other"

    def save(self, content: str, category: str | None = None) -> dict[str, Any]:
        module = self._ensure_module()
        final_category = category or self._classify_category(content)
        item = self._call_quietly(module.save, content, final_category)
        return {
            "ok": True,
            "action": "save",
            "category": final_category,
            "item": item,
            "message": f"已记住：{content}",
        }

    def search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        module = self._ensure_module()
        results = self._call_quietly(module.search, query, top_k)
        return {
            "ok": True,
            "action": "search",
            "query": query,
            "results": results,
            "message": f"找到 {len(results)} 条相关记忆" if results else "没有找到相关记忆",
        }

    def delete(self, query: str) -> dict[str, Any]:
        module = self._ensure_module()
        deleted = self._call_quietly(module.delete_by_query, query)
        return {
            "ok": bool(deleted),
            "action": "delete",
            "query": query,
            "message": f"已删除与“{query}”最相关的记忆" if deleted else f"没有找到与“{query}”相关的记忆",
        }

    def list_all(self, category: str | None = None) -> dict[str, Any]:
        module = self._ensure_module()
        results = self._call_quietly(module.list_all, category)
        return {
            "ok": True,
            "action": "list",
            "category": category,
            "results": results,
            "message": f"当前共有 {len(results)} 条记忆",
        }

    def search_for_injection(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        try:
            result = self.search(query, top_k=top_k)
        except Exception:
            return []
        return [item for item in result.get("results", []) if isinstance(item, dict)]
