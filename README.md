# ACEE Multi-Agent CLI

一个基于 **Textual TUI + Orchestrator + Shell/Tool/Memory Agent** 的命令行多 Agent 系统。

## 依赖安装

推荐使用虚拟环境：

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

如 `requirements.txt` 中缺少 MCP 相关包，可补装：

```bash
pip install mcp
```

## 环境配置

项目启动前需要配置 `.env`，至少保证可创建 OpenAI 兼容客户端，例如：

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=your_base_url
OPENAI_MODEL=gpt-5
```

`OPENAI_MODEL` 可按你的网关实际支持情况调整。

如需跳过本地 Ollama Fast-Path、固定使用云端意图路由，可在 `.env` 中添加：

```env
LOCAL_FASTPATH_ENABLED=false
```

也可以使用等价开关：

```env
FORCE_CLOUD_ROUTING=true
```

## 运行方式

### 方式 1：使用启动脚本

```bash
bash start_agent.sh
```

默认工作区是执行脚本时所在目录。也可以显式指定工作区：

```bash
bash start_agent.sh --workspace /path/to/workspace
AGENT_WORKSPACE=/path/to/workspace bash start_agent.sh
```

如果 Python 不在默认 `venv` 中：

```bash
PYTHON_BIN=/path/to/python bash start_agent.sh
```

### 方式 2：安装命令标识符

可把仓库内包装器软链到 `PATH` 中的目录，获得类似 `codex` / `claude` 的启动方式：

```bash
mkdir -p ~/.local/bin
ln -sf "$(pwd)/bin/acee-agent" ~/.local/bin/acee-agent
```

之后可在任意工作区运行：

```bash
cd /path/to/workspace
acee-agent
acee-agent --workspace /other/workspace
```

### 方式 3：直接运行入口

```bash
export PYTHONPATH=$(pwd)
python -m src.main
```

## 启动检查

可先检查启动脚本解析是否正常：

```bash
bash start_agent.sh --check
```
