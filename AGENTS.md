# Repository Guidelines

## 项目结构与模块组织

本仓库是一个基于 Textual TUI 的 Python 多 Agent 命令行系统。入口文件是 `src/main.py`。核心编排逻辑位于 `src/orchestrator/`，Shell 命令规划与执行位于 `src/shell_agent/`，MCP 工具、工具客户端与权限策略位于 `src/tool_agent/`，终端界面组件位于 `src/tui/`。`Bonus/` 存放 A2A 通信、本地 NLP 路由、记忆系统和会话管理等扩展模块。`docs/` 放架构说明和接口文档。`Pretask1/`、`pre/`、`test/` 主要用于 HTML/CSS/JS 示例或生成结果。

## 构建、测试与本地运行

首次开发建议创建虚拟环境并安装依赖：

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

启动主程序：

```bash
bash start_agent.sh
```

检查启动脚本解析：

```bash
bash start_agent.sh --check
```

若 Python 不在默认 `venv` 中，使用：

```bash
PYTHON_BIN=/path/to/python bash start_agent.sh
```

提交前至少运行语法检查：

```bash
venv/bin/python -m compileall -q src Bonus examples Pretask2
```

## 编码风格与命名约定

Python 代码使用 4 空格缩进，优先补充清晰的类型标注和显式错误处理。模块名保持小写加下划线。保持边界清晰：路由逻辑放在 `orchestrator`，命令执行放在 `shell_agent`，MCP 工具和权限判断放在 `tool_agent`。工具返回值优先使用结构化 JSON；面向用户的中文提示应简短、明确。

## 测试要求

当前仓库没有完整 pytest 测试套件，`compileall` 是最低验证要求。修改 MCP 工具时，应增加或运行函数级冒烟脚本，覆盖合法路径、非法路径、权限分支和返回字段。生成的临时 demo 或测试文件应放在 `test/` 或被 `.gitignore` 忽略的临时目录中。

## 提交与 Pull Request 规范

近期提交信息多为简短中文摘要，例如 `清理 Git 跟踪`、`欢迎界面 + Prompt工程优化`。继续使用这种风格：简洁、面向结果、范围明确。不要自动执行 `git commit` 或 `git push`；只有在用户明确要求提交或推送时才执行。PR 应说明用户可见行为、涉及模块、配置项变化和验证命令；TUI 相关改动建议附截图或终端输出。

## 安全与配置提示

不要提交 `.env`、记忆 JSON、会话记录、虚拟环境、日志或临时输出。所有工具路径处理应限制在当前工作目录内的相对路径；写入、替换、删除等有副作用操作必须保留确认流程。
