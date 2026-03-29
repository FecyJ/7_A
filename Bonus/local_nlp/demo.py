"""本地 NLP Bonus 演示脚本。"""

from __future__ import annotations

import asyncio
import json
import sys

try:
    from .router import route_user_input
except ImportError:
    from router import route_user_input  # type: ignore


async def main() -> None:
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:]).strip()
    else:
        user_input = input(">>> ").strip()

    if not user_input:
        print("请输入一条自然语言指令。")
        return

    result = await route_user_input(user_input, status_callback=lambda msg: print(f"[status] {msg}"))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
