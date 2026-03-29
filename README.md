## 0. 架构愿景与重构目标 (Vision & Objectives)
本项目正处于从“单体脚本”向“分布式多智能体框架”蜕变的关键阶段。
本次重构包含两个循序渐进的阶段：
1. **Phase 1: A2A (Agent-to-Agent) 底层通信总线重构**。将现有的硬编码函数调用彻底剥离，建立基于标准消息传递（Message-Passing）的通信协议，实现 Agent 的即插即用。
2. **Phase 2: 读写分离的混合记忆架构 (Hybrid Memory Architecture)**。基于团队已实现的 JSON 本地记忆库（包含 jieba 检索、0.4 冲突阈值等特性），接入短期 RAM 与长期 ROM，彻底解决大模型的“失忆症”。

---

## 🚀 Phase 1: A2A 协议改造 (The Message Bus)

**【严格遵守】 为了保证核心业务逻辑的纯粹性，以及功能模块的清晰划分，所有 A2A 协议的基础设施代码必须实现在特定的 Bonus/a2a/ 目录下。**

请在该目录下创建核心协议文件（例如 Bonus/a2a/protocol.py、Bonus/a2a/registry.py 等）。

原有的业务 Agent 代码（Shell, Tool 等）不允许把基类定义写在自己的文件里，必须通过 from Bonus.a2a.protocol import A2AMessage, BaseAgent 的方式引入

### 1. 核心概念：标准信封与统一接口 (Contracts)
系统内所有 Agent 之间的交互，必须遵循统一的“信封”格式和处理接口。
* **消息实体 (`A2AMessage`)**: 设计一个标准的数据结构（建议基于 Pydantic）。必须包含：唯一标识 (ID)、发送方 (Sender)、接收方 (Receiver)、消息类型 (Intent/Type)、核心载荷 (Payload) 以及上下文环境 (Context)。
* **基类契约 (`BaseAgent`)**: 抽象出一个 Agent 基类。无论是 Shell Agent 还是 Tool Agent，都必须实现一个统一的异步入口方法（例如 `process_message`）。该方法必须接收一个 `A2AMessage` 请求，并返回一个 `A2AMessage` 响应。

### 2. 消息总线与动态路由 (Dynamic Routing)
* **废除硬编码**: 移除主控（Orchestrator）中类似 `if intent == "shell": await shell_agent.run()` 的硬耦合代码。
* **引入注册表 (`AgentRegistry`)**: 实现一个中心化的注册机制。系统启动时，将各个子 Agent 实例注册进去。主控逻辑只需提取 LLM 输出的 `intent`，从注册表中获取目标 Agent，并派发标准化消息。

---

## 🧠 Phase 2: 读写分离的混合记忆接入 (CQRS Hybrid Memory)

本阶段要求将上下文管理分为“短期工作记忆 (RAM)”与“长期持久化记忆 (ROM)”，并采用类似 CQRS（命令查询职责分离）的模式进行接入。必须复用团队成员已写好的基于 JSON 和 jieba 分词的记忆引擎。

### 1. 短期记忆 (RAM): 会话上下文维护
* **位置**: 驻留在 Orchestrator（总控层）或主循环生命周期内。
* **机制**: 维护一个滑动窗口式的历史对话队列。每次向云端/本地大模型发起请求时，将此队列作为上下文拼接进 `messages` 数组。响应后，将新一轮对话追加进队列。
* **目的**: 解决多轮对话中的“代词指代”问题（如“把**那个**文件删了”）。

### 2. 长期记忆 (ROM) 的读操作：前置 RAG 注入 (Context Injector)
* **时机**: 在 Orchestrator 向大模型发起思考请求**之前**进行。
* **流程**: 
    1. 拦截用户的原始自然语言输入。
    2. 调用团队已实现的 jieba 检索模块，在 JSON 记忆库中搜索高匹配度（Top-K）的记忆片段。
    3. 将检索到的记忆作为“背景事实（Background Facts）”或“用户偏好（Preferences）”，无缝拼接（Inject）到 Orchestrator 的 System Prompt 中。
* **目的**: 让大模型在规划任务时，预先知道诸如“用户的默认开发语言”、“数据库的密码”等持久化信息。

### 3. 长期记忆 (ROM) 的写操作：独立的 A2A 节点 (`MemoryAgent`)
* **定位**: 将记忆的增删改包装为一个符合 Phase 1 标准的独立子 Agent。
* **流程**:
    1. 当 Orchestrator 识别到用户有“存储信息”的意图时（例如“记住我的开发服务器 IP 是 xxx”），生成一个指向 `memory_agent` 的 A2A 任务包。
    2. `MemoryAgent` 接收包后，提取 Payload。
    3. 调用团队已实现的写入逻辑：执行关键词冲突检测（利用现有的动态阈值或 0.4 阈值设计）。如果冲突则更新旧记忆，否则创建新记忆，持久化保存到 JSON 文件中。
    4. 返回 A2A 成功响应。

---

## 🔄 核心数据流运转示例 (Data Flow Trace)

请 AI 助手在重构时，确保系统能够顺畅执行以下链路：
1. 用户输入：*“用我最喜欢的语言写一个 Hello World，并记住这个文件路径。”*
2. **ROM 读取**: RAG 模块触发，从 JSON 查出*“最喜欢的语言是 Python”*，注入给总控。
3. **RAM 提取**: 携带当前终端工作目录和前几轮对话。
4. **总控思考**: 双擎路由（本地 Ollama 或云端 API）得出两步操作意图。
5. **A2A 派发 1**: 封装写代码任务发给 `ShellAgent/ToolAgent`，等待返回。
6. **A2A 派发 2**: 封装存储路径任务发给 `MemoryAgent`，等待 JSON 库更新。
7. **RAM 更新**: 将本次交互存入滑动窗口，返回结果给 TUI 界面。

## ⚠️ 工程约束 (Engineering Constraints)
* **平滑过渡**: 优先搭建 A2A 基类和注册表，将现有的 Shell 和 Tool 逻辑原封不动地包进去，确保单元测试通过后，再剥离 Orchestrator 的硬编码。
* **尊重历史实现**: 尽量不要重写组员在 Bonus 1 中完成的 `jieba` 分词检索和 `JSON` CRUD 逻辑，将其作为黑盒 Service 导入并调用即可。但记忆触发方式可扩充化，尝试能实现隐式自然语言触发

***