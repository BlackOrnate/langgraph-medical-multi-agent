# LangGraph Medical Multi-Agent

**English** | [简体中文](README.zh-CN.md)

A learning project for building a **supervisor-style multi-agent system** with [LangGraph](https://github.com/langchain-ai/langgraph). A supervisor classifies the user's question, routes it to a specialist agent, then summarizes that agent's result into the final reply.

Each specialist demonstrates a different technique:

| Agent | Handles | Technique |
| --- | --- | --- |
| `history` | The user's own visit records | Tool calling + **MCP** server, with `user_id` injected through runtime context |
| `guide` | Symptom questions | **RAG** over a Markdown guideline with a **FAISS** vector store |
| `doctor` | Doctor departments, titles, clinic hours | **Text-to-SQL** on an in-memory SQLite database |
| `reject` | Anything unrelated to healthcare | Polite refusal |

Everything runs locally through [Ollama](https://ollama.com). LangSmith tracing is optional.

> ⚠️ All medical content in this repo is mock data for learning. It is not medical advice.

## Architecture

```text
START -> supervisor_node (classify)
           ├─ history -> history_node  (Tools + MCP)     ─┐
           ├─ guide   -> guide_node    (RAG + FAISS)     ─┤
           ├─ doctor  -> doctor_node   (Text-to-SQL)     ─┼-> supervisor_node (summarize) -> END
           └─ other   -> reject_node   (polite refusal)  ─┘
```

The supervisor node runs twice. On the first pass it classifies the question with structured output (`RouteDecision`). When an agent returns, `agent_result` is set in the state, so the supervisor summarizes it and routes to `END`.

## Project Structure

```text
medical_multi_agent/
├── config.py            Model names, file paths, MCP address
├── state.py             Shared graph State
├── mcp_server.py        Medical records MCP server (port 8001)
├── agents/
│   ├── common.py        Shared helpers: create model, call agent, node return value
│   ├── supervisor.py    Classification + summary + routing function
│   ├── history.py       Visit history agent (queries through MCP)
│   ├── guide.py         Medical guideline agent (RAG)
│   ├── doctor.py        Doctor information agent (generates SQL)
│   └── reject.py        Polite refusal
├── graph.py             Builds the StateGraph
├── main.py              Command-line runner
├── app.py               Gradio demo UI
└── data/
    └── medical_guideline.md   Care guideline (mock data)
```

## Getting Started

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) running locally, with these models pulled:

```bash
ollama pull gemma4:e4b
ollama pull nomic-embed-text-v2-moe:latest
```

To use different models, change `CHAT_MODEL` and `EMBEDDING_MODEL` in `config.py`.

### Install

```bash
pip install -r requirements.txt
```

### Optional: LangSmith tracing

```bash
cp .env.example .env   # then fill in your LANGSMITH_API_KEY
```

### Run

```bash
# Terminal 1: start the MCP server (keep it running)
python mcp_server.py

# Terminal 2: pick one
python main.py        # run the four sample questions
python main.py -i     # ask questions interactively
python app.py         # Gradio UI at http://127.0.0.1:7860
```

On first run, the vector store `faiss_index_medical/` is built from `data/medical_guideline.md`. If you edit the guideline, delete that folder and run again to rebuild it.

## Mock Data

| Data | Location | Content |
| --- | --- | --- |
| Visit records | `mcp_server.py` | Records for users 1 and 2; user 99 has none |
| Care guideline | `data/medical_guideline.md` | 7 common conditions |
| Doctors | `agents/doctor.py` | 8 doctors with department, title and clinic hours |

## Sample Questions

- `Show me my previous medical visits` → `history`
- `I've been coughing for three days and have a slight fever. What should I do?` → `guide`
- `Which Cardiology doctors are available on Wednesday?` → `doctor`
- `Will the stock market go up today?` → `other`
