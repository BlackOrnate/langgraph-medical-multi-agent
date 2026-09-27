# LangGraph 医疗问诊多智能体

[English](README.md) | **简体中文**

一个使用 [LangGraph](https://github.com/langchain-ai/langgraph) 构建 **Supervisor 模式多智能体系统** 的学习项目。Supervisor 对用户问题进行分类，路由到对应的专业 Agent 处理，最后把 Agent 的处理结果汇总成最终回复。

每个专业 Agent 演示一种不同的技术：

| Agent | 负责内容 | 技术 |
| --- | --- | --- |
| `history` | 用户自己的就诊记录 | 工具调用 + **MCP** Server，`user_id` 通过运行时上下文注入 |
| `guide` | 症状、病情咨询 | 基于 Markdown 指南的 **RAG**，使用 **FAISS** 向量库 |
| `doctor` | 医生科室、职称、出诊时间 | 在内存 SQLite 数据库上做 **Text-to-SQL** |
| `reject` | 与医疗无关的问题 | 礼貌拒绝 |

所有模型都通过 [Ollama](https://ollama.com) 在本地运行，LangSmith 追踪为可选项。

> ⚠️ 本仓库中的所有医疗内容均为教学用模拟数据，不构成任何医疗建议。
>
> 项目中的代码、界面和数据均为英文，向模型提问时也建议使用英文。

## 架构

```text
START -> supervisor_node（分类）
           ├─ history -> history_node  (Tools + MCP)     ─┐
           ├─ guide   -> guide_node    (RAG + FAISS)     ─┤
           ├─ doctor  -> doctor_node   (Text-to-SQL)     ─┼-> supervisor_node（汇总） -> END
           └─ other   -> reject_node   (礼貌拒绝)         ─┘
```

Supervisor 节点会经过两次。第一次通过结构化输出（`RouteDecision`）对问题分类；Agent 处理完返回后，State 中已有 `agent_result`，Supervisor 就对结果进行汇总，然后路由到 `END`。

## 目录结构

```text
medical_multi_agent/
├── config.py            模型名称、文件路径、MCP 地址
├── state.py             全局 State
├── mcp_server.py        就诊记录 MCP Server（端口 8001）
├── agents/
│   ├── common.py        公共方法：创建模型、调用 Agent、节点返回值
│   ├── supervisor.py    分类 + 汇总 + 路由函数
│   ├── history.py       历史就诊记录 Agent（通过 MCP 查询）
│   ├── guide.py         诊疗指南 Agent（RAG）
│   ├── doctor.py        医生信息 Agent（生成 SQL 查询）
│   └── reject.py        礼貌拒绝
├── graph.py             组装 StateGraph
├── main.py              命令行运行
├── app.py               Gradio 演示界面
└── data/
    └── medical_guideline.md   诊疗指南（模拟数据）
```

## 快速开始

### 前提条件

- Python 3.11+
- 本地已启动 [Ollama](https://ollama.com)，并已下载以下模型：

```bash
ollama pull gemma4:e4b
ollama pull nomic-embed-text-v2-moe:latest
```

如需更换模型，修改 `config.py` 中的 `CHAT_MODEL` 和 `EMBEDDING_MODEL` 即可。

### 安装依赖

```bash
pip install -r requirements.txt
```

### 可选：LangSmith 追踪

```bash
cp .env.example .env   # 然后填入你自己的 LANGSMITH_API_KEY
```

### 运行

```bash
# 终端 1：启动 MCP Server（保持运行）
python mcp_server.py

# 终端 2：任选一种方式运行
python main.py        # 运行四个示例问题
python main.py -i     # 命令行交互式提问
python app.py         # Gradio 界面，打开 http://127.0.0.1:7860
```

第一次运行时会根据 `data/medical_guideline.md` 生成向量库 `faiss_index_medical/`。修改指南文档后，删除该文件夹再运行即可重新生成。

## 模拟数据

| 数据 | 位置 | 内容 |
| --- | --- | --- |
| 就诊记录 | `mcp_server.py` | 用户 1、用户 2 的就诊记录；用户 99 没有记录 |
| 诊疗指南 | `data/medical_guideline.md` | 7 种常见病症 |
| 医生信息 | `agents/doctor.py` | 8 位医生的科室、职称、出诊时间 |

## 示例问题

- `Show me my previous medical visits`（查询就诊记录）→ `history`
- `I've been coughing for three days and have a slight fever. What should I do?`（咳嗽发烧怎么办）→ `guide`
- `Which Cardiology doctors are available on Wednesday?`（心内科周三有哪些医生出诊）→ `doctor`
- `Will the stock market go up today?`（今天股票会涨吗）→ `other`
