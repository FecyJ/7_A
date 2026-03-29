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

## 运行方式

### 方式 1：使用启动脚本

```bash
bash start_agent.sh
```

如果 Python 不在默认 `venv` 中：

```bash
PYTHON_BIN=/path/to/python bash start_agent.sh
```

### 方式 2：直接运行入口

```bash
export PYTHONPATH=$(pwd)
python -m src.main
```

## 启动检查

可先检查启动脚本解析是否正常：

```bash
bash start_agent.sh --check
```

