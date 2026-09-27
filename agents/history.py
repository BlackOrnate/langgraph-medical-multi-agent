"""Medical history Agent: queries the current user's visit records through the MCP Server."""
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware
from langchain.tools import tool, ToolRuntime
from langchain_mcp_adapters.client import MultiServerMCPClient

from agents.common import agent_update, ask_agent, create_model
from config import MCP_URL
from state import State


mcp_client = MultiServerMCPClient(
    {"medical": {"url": MCP_URL, "transport": "streamable_http"}}
)
_mcp_query_records = None


async def get_mcp_query_records():
    """Connect to the MCP Server and fetch the tool on first use, then reuse it."""
    global _mcp_query_records
    if _mcp_query_records is None:
        mcp_tools = await mcp_client.get_tools()
        _mcp_query_records = next(t for t in mcp_tools if t.name == "query_medical_records")
    return _mcp_query_records


# Runtime context: user_id is passed in by the program, not filled in by the model
@dataclass
class UserContext:
    user_id: int


@tool
async def query_medical_records(runtime: ToolRuntime[UserContext]) -> str:
    """Query the current user's medical visit history, sorted from most recent to oldest."""
    mcp_tool = await get_mcp_query_records()
    # Call the MCP tool with the user_id passed in by the program, so the model can't query other users
    result = await mcp_tool.ainvoke({"user_id": runtime.context.user_id})
    # The MCP tool returns a list of content blocks: [{"type": "text", "text": "..."}]
    return "\n".join(block["text"] for block in result if block["type"] == "text")


HISTORY_FAIL_TEXT = "Sorry, your medical records are temporarily unavailable. Please try again later or ask the hospital information desk."

history_agent = create_agent(
    model=create_model(),
    tools=[query_medical_records],
    system_prompt="""
        You are the hospital's medical records assistant. You answer questions about the user's own visit history,
        such as what conditions they were treated for, which departments they visited, and what medications were prescribed.

        First call query_medical_records to get the current user's records, then answer based on the results.
        Records are sorted from most recent to oldest, so "last time" / "most recent visit" refers to the first record.

        Answer only from the query results and never make up records; if there are no records, tell the user honestly.
    """,
    context_schema=UserContext,
    middleware=[ModelRetryMiddleware(max_retries=2)],
)


async def history_node(state: State):
    print("~~~~~~This is History Agent~~~~~~")
    answer = await ask_agent(
        history_agent,
        state["messages"][0].content,
        HISTORY_FAIL_TEXT,
        # Pass the user_id from State to the tool
        context=UserContext(user_id=state.get("user_id")),
    )
    return agent_update(answer)
