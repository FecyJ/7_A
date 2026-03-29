"""
Bonus 1 - Memory Agent（跨会话持久化记忆）
功能：save / search / update / delete / list_all
存储：JSON 文件，支持增删改查
检索：jieba 分词 + 关键词匹配度排序
"""

import os
import json
import uuid
import jieba
from datetime import datetime

# 记忆存储文件路径
MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.json")

# 记忆分类
CATEGORIES = {
    "project": "项目信息",
    "preference": "用户偏好",
    "fact": "事实信息",
    "other": "其他"
}

# 超过此数量提醒用户清理
MEMORY_WARN_THRESHOLD = 50

# jieba 停用词（常见无意义词）
STOP_WORDS = {
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "被",
    "从", "把", "对", "用", "这个", "那个", "什么", "怎么", "可以", "能",
    "吗", "呢", "吧", "啊", "哦", "嗯", "过", "还", "但", "而", "或",
    "如果", "因为", "所以", "但是", "然后", "已经", "正在", "非常", "比较"
}


def _load_memory() -> dict:
    """从 JSON 文件加载记忆数据"""
    if not os.path.exists(MEMORY_FILE):
        return {
            "_meta": {
                "total": 0,
                "last_updated": "",
                "tip": "这是 Memory Agent 的持久化存储文件，可以手动编辑。每条记忆包含 id、category、content、keywords、timestamp 字段。"
            },
            "memories": []
        }
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_memory(data: dict):
    """将记忆数据写入 JSON 文件"""
    data["_meta"]["total"] = len(data["memories"])
    data["_meta"]["last_updated"] = datetime.now().isoformat()
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _extract_keywords(text: str) -> list:
    """用 jieba 分词提取关键词（去停用词、去单字）"""
    words = jieba.lcut(text)
    keywords = [w.strip() for w in words if w.strip() and w not in STOP_WORDS and len(w) > 1]
    return list(set(keywords))  # 去重


def _match_score(query_keywords: list, memory_keywords: list) -> float:
    """计算 query 和 memory 之间的关键词匹配度"""
    if not query_keywords:
        return 0.0
    matched = sum(1 for kw in query_keywords if kw in memory_keywords)
    return matched / len(query_keywords)


# === 核心功能 ===

def save(content: str, category: str = "other") -> dict:
    """
    保存一条记忆。
    自动提取关键词，检测冲突记忆（关键词重叠度 >= 0.6 视为冲突，更新而非新增）。
    返回保存/更新的记忆条目。
    """
    data = _load_memory()
    keywords = _extract_keywords(content)

    # 冲突检测：检查是否有高度重叠的旧记忆
    # fact 分类（事实信息）用更高阈值，避免误覆盖不同的事实
    for mem in data["memories"]:
        overlap = _match_score(keywords, mem["keywords"])
        threshold = 0.7 if mem["category"] == "fact" else 0.4
        if overlap >= threshold:
            # 冲突 → 更新旧记忆
            old_content = mem["content"]
            mem["content"] = content
            mem["keywords"] = keywords
            mem["category"] = category
            mem["timestamp"] = datetime.now().isoformat()
            _save_memory(data)

            # 检查数量提醒
            warn = ""
            if len(data["memories"]) >= MEMORY_WARN_THRESHOLD:
                warn = f"\n[提醒] 当前已有 {len(data['memories'])} 条记忆，建议清理不需要的条目"

            print(f"[已更新] 原记忆: {old_content}")
            print(f"         新记忆: {content}（分类：{CATEGORIES.get(category, category)}）{warn}")
            return mem

    # 无冲突 → 新增
    new_memory = {
        "id": str(uuid.uuid4())[:8],
        "category": category,
        "content": content,
        "keywords": keywords,
        "timestamp": datetime.now().isoformat()
    }
    data["memories"].append(new_memory)
    _save_memory(data)

    # 检查数量提醒
    warn = ""
    if len(data["memories"]) >= MEMORY_WARN_THRESHOLD:
        warn = f"\n[提醒] 当前已有 {len(data['memories'])} 条记忆，建议清理不需要的条目"

    print(f"[已记住] {content}（分类：{CATEGORIES.get(category, category)}）{warn}")
    return new_memory


def search(query: str, top_k: int = 5) -> list:
    """
    根据 query 检索相关记忆。
    用 jieba 分词提取关键词，按匹配度排序，返回 Top-K。
    匹配度为 0 的不返回。
    """
    data = _load_memory()
    if not data["memories"]:
        print("[记忆] 当前没有任何记忆")
        return []

    query_keywords = _extract_keywords(query)
    if not query_keywords:
        print("[记忆] 无法从查询中提取关键词")
        return []

    # 计算每条记忆的匹配度
    scored = []
    for mem in data["memories"]:
        score = _match_score(query_keywords, mem["keywords"])
        if score > 0:
            scored.append((score, mem))

    # 按匹配度降序排列
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [mem for _, mem in scored[:top_k]]

    # 友好展示
    if not results:
        print("[记忆] 没有找到相关记忆")
    else:
        print(f"\n找到 {len(results)} 条相关记忆：")
        for mem in results:
            cat_name = CATEGORIES.get(mem["category"], mem["category"])
            date = mem["timestamp"][:10]
            print(f"  [{cat_name}] {mem['content']} ({date}) [id: {mem['id']}]")
        print()

    return results


def delete(memory_id: str) -> bool:
    """根据 id 删除一条记忆"""
    data = _load_memory()
    for i, mem in enumerate(data["memories"]):
        if mem["id"] == memory_id:
            removed = data["memories"].pop(i)
            _save_memory(data)
            print(f"[已删除] {removed['content']}")
            return True
    print(f"[错误] 未找到 id 为 {memory_id} 的记忆")
    return False


def delete_by_query(query: str) -> bool:
    """根据关键词匹配删除最相关的一条记忆"""
    data = _load_memory()
    query_keywords = _extract_keywords(query)
    if not query_keywords:
        print("[错误] 无法从输入中提取关键词")
        return False

    best_score = 0
    best_idx = -1
    for i, mem in enumerate(data["memories"]):
        score = _match_score(query_keywords, mem["keywords"])
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx >= 0 and best_score > 0:
        removed = data["memories"].pop(best_idx)
        _save_memory(data)
        print(f"[已删除] {removed['content']}")
        return True
    else:
        print("[记忆] 没有找到匹配的记忆")
        return False


def list_all(category: str = None) -> list:
    """列出所有记忆，可按分类筛选"""
    data = _load_memory()
    memories = data["memories"]

    if category:
        memories = [m for m in memories if m["category"] == category]

    if not memories:
        print("[记忆] 当前没有记忆" + (f"（分类：{category}）" if category else ""))
        return []

    # 按分类分组展示
    grouped = {}
    for mem in memories:
        cat = mem["category"]
        if cat not in grouped:
            grouped[cat] = []
        grouped[cat].append(mem)

    print(f"\n共 {len(memories)} 条记忆：")
    for cat, mems in grouped.items():
        cat_name = CATEGORIES.get(cat, cat)
        print(f"\n  【{cat_name}】")
        for mem in mems:
            date = mem["timestamp"][:10]
            print(f"    - {mem['content']} ({date}) [id: {mem['id']}]")
    print()

    return memories


def clear() -> int:
    """清空所有记忆，返回删除的条目数"""
    data = _load_memory()
    count = len(data["memories"])
    data["memories"] = []
    _save_memory(data)
    print(f"[已清空] 共删除 {count} 条记忆")
    return count


# === 自然语言触发识别 ===

def detect_memory_intent(user_input: str) -> tuple[str, str]:
    """
    检测用户输入是否为记忆相关指令。
    返回 (action, content)：
      - ("save", 要保存的内容)
      - ("search", 查询内容)
      - ("delete", 要删除的内容)
      - ("list", "")
      - (None, "") 表示不是记忆指令
    """
    text = user_input.strip()

    # 保存类
    for trigger in ["记住", "记下", "保存记忆", "帮我记"]:
        if trigger in text:
            content = text.split(trigger, 1)[-1].strip()
            return ("save", content) if content else ("save", text)

    # 检索类
    for trigger in ["你记得", "记得吗", "之前说过", "我说过什么", "你还记得", "回忆一下"]:
        if trigger in text:
            content = text.replace(trigger, "").strip()
            return ("search", content) if content else ("search", text)

    # 删除类
    for trigger in ["忘掉", "忘记", "删除记忆", "去掉记忆"]:
        if trigger in text:
            content = text.split(trigger, 1)[-1].strip()
            return ("delete", content) if content else (None, "")

    # 列出类
    for trigger in ["所有记忆", "列出记忆", "查看记忆", "有哪些记忆"]:
        if trigger in text:
            return ("list", "")

    return (None, "")


def handle_memory_command(user_input: str, classify_func=None) -> bool:
    """
    处理记忆相关指令。
    classify_func: 用于自动分类的函数（传入内容，返回 category 字符串）
    返回 True 表示已处理（是记忆指令），False 表示不是记忆指令。
    """
    action, content = detect_memory_intent(user_input)

    if action is None:
        return False

    if action == "save":
        # 自动分类
        category = "other"
        if classify_func:
            category = classify_func(content)
        save(content, category)

    elif action == "search":
        search(content)

    elif action == "delete":
        delete_by_query(content)

    elif action == "list":
        list_all()

    return True


# === LLM 自动分类 ===

def auto_classify_category(content: str, llm_client=None, model: str = None) -> str:
    """
    调用 LLM 自动判断记忆的分类。
    返回 category 字符串：project / preference / fact / other
    """
    if llm_client is None:
        return "other"

    prompt = f"""请判断以下内容属于哪个分类，只回复分类名称（一个英文单词）：

分类选项：
- project（项目相关：框架、技术栈、依赖、架构等）
- preference（用户偏好：习惯、风格、设置等）
- fact（事实信息：人名、日期、数字、规则等）
- other（不属于以上分类）

内容：{content}

只回复分类名称："""

    try:
        response = llm_client.chat.completions.create(
            model=model or "claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        result = response.choices[0].message.content.strip().lower()
        if result in CATEGORIES:
            return result
    except Exception:
        pass
    return "other"


# === 主程序（独立测试用）===
if __name__ == "__main__":
    print("=" * 50)
    print("  Memory Agent 测试")
    print("=" * 50)

    # 清空旧数据
    clear()

    # 保存几条记忆
    save("这个项目使用 Flask 框架开发", "project")
    save("团队成员有张三、李四、王五", "fact")
    save("用户喜欢简洁的代码风格", "preference")

    # 搜索
    print("\n--- 搜索测试 ---")
    search("项目用什么框架")
    search("团队有谁")

    # 冲突检测
    print("\n--- 冲突检测测试 ---")
    save("这个项目改用 Django 框架了", "project")

    # 列出所有
    print("\n--- 列出所有记忆 ---")
    list_all()

    # 删除
    print("\n--- 删除测试 ---")
    delete_by_query("团队成员")

    # 最终状态
    print("\n--- 最终状态 ---")
    list_all()
