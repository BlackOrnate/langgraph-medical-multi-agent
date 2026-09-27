"""Supervisor: classifies the question on the first pass, then summarizes once an Agent returns."""
from typing import Literal

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware
from langgraph.graph import END
from pydantic import BaseModel, Field

from agents.common import ask_agent, create_model
from state import State


# Category -> next node name
CATEGORY_TO_NODE = {
    "history": "history_node",
    "guide": "guide_node",
    "doctor": "doctor_node",
    "other": "reject_node",
}


# Structured output: the model can only return one of these categories
class RouteDecision(BaseModel):
    category: Literal["history", "guide", "doctor", "other"] = Field(
        description="Category of the user's question"
    )
    reason: str = Field(description="Brief reason for the classification")


# Classification needs stable output, so temperature is 0
router_agent = create_agent(
    model=create_model(),
    system_prompt="""
        You are the triage assistant of a medical consultation system. Classify the user's question into one of these categories:

        - history: the user's own visit history, medical records, previous diagnoses or medications
        - guide: describing symptoms, asking about a condition, possible diseases or treatment advice
        - doctor: doctor information, such as department, title, clinic hours, or doctor recommendations
        - other: questions unrelated to healthcare
    """,
    # Structured output; the result is stored in response["structured_response"]
    response_format=RouteDecision,
)

# Summarizing only reorganizes existing content and needs no creativity, so temperature is 0
summary_agent = create_agent(
    model=create_model(),
    system_prompt="""
        You are the reply-writing assistant of the hospital's consultation system.
        You will receive the user's question and the result from a specialist assistant. Turn that result into the final reply:

        1. Use a friendly, warm tone and respond directly to the user's question
        2. Keep it well structured; use bullet points when there is a lot of content
        3. Only reorganize information already in the result; do not add, remove or change any facts (dates, departments, doctors, medications, etc.)
        4. If the result contains a "for reference only" style disclaimer, you must keep it
    """,
    middleware=[ModelRetryMiddleware(max_retries=2)],
)


def route_by_category(state: State):
    # Summary is done, end the flow
    if state.get("final_answer"):
        return END
    return CATEGORY_TO_NODE.get(state.get("category"), "reject_node")


async def classify(question: str) -> str:
    try:
        response = await router_agent.ainvoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        decision = response["structured_response"]
        print(f"Category: {decision.category}, reason: {decision.reason}")
        return decision.category
    except Exception as e:
        # Fall back to "other" if the Agent call or structured output fails
        print(f"Classification failed, defaulting to other: {e}")
        return "other"


async def summarize(state: State) -> str:
    question = state["messages"][0].content
    agent_result = state.get("agent_result", "")

    # The polite refusal is already short, no need to summarize it
    if state.get("category") == "other":
        return agent_result

    # If summarizing fails, use the Agent's result directly
    return await ask_agent(
        summary_agent,
        f"User question: {question}\n\nResult:\n{agent_result}",
        fail_text=agent_result,
    )


async def supervisor_node(state: State):
    print("~~~~~~This is Supervisor~~~~~~")

    # Back at the supervisor after an Agent finished: summarize the result
    if state.get("agent_result"):
        final_answer = await summarize(state)
        print(final_answer)
        return {"final_answer": final_answer, "messages": [{"role": "ai", "content": final_answer}]}
    else:
        # First pass through the supervisor: classify the user's question
        category = await classify(state["messages"][0].content)
        return {"category": category}
