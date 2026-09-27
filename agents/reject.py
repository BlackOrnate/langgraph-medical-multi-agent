"""Other questions: politely decline."""
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware

from agents.common import agent_update, ask_agent, create_model
from state import State


REJECT_TEXT = (
    "Sorry, I'm a medical consultation assistant and can only answer health-related questions, "
    "such as looking up your visit history, asking about symptoms, or finding doctor information. "
    "Is there anything health-related I can help you with?"
)

reject_agent = create_agent(
    # Allow a little variation in the refusal so it reads more naturally
    model=create_model(temperature=0.3),
    system_prompt="""
        You are a medical consultation assistant and only handle health-related questions,
        including: looking up visit history, asking about symptoms, and finding doctor information.

        The user's question is outside your scope. Politely decline in one or two sentences and guide the user toward a health-related question.
        Do not answer the content of the user's question itself.
    """,
    middleware=[ModelRetryMiddleware(max_retries=2)],
)


async def reject_node(state: State):
    print("~~~~~~This is Reject Agent~~~~~~")
    answer = await ask_agent(reject_agent, state["messages"][0].content, REJECT_TEXT)
    return agent_update(answer)
