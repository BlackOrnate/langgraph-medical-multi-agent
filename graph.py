from langgraph.graph import END, START, StateGraph

from agents.doctor import doctor_node
from agents.guide import guide_node
from agents.history import history_node
from agents.reject import reject_node
from agents.supervisor import CATEGORY_TO_NODE, route_by_category, supervisor_node
from state import State


def build_graph():
    builder = StateGraph(State)
    builder.add_node("supervisor_node", supervisor_node)
    builder.add_node("history_node", history_node)
    builder.add_node("guide_node", guide_node)
    builder.add_node("doctor_node", doctor_node)
    builder.add_node("reject_node", reject_node)

    # START -> supervisor classifies -> Agent handles -> supervisor summarizes -> END
    builder.add_edge(START, "supervisor_node")
    builder.add_conditional_edges(
        "supervisor_node", route_by_category, [*CATEGORY_TO_NODE.values(), END]
    )
    for node in CATEGORY_TO_NODE.values():
        builder.add_edge(node, "supervisor_node")

    return builder.compile()


graph = build_graph()
