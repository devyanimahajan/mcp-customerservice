# workflows/a2a_graph.py

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage

from agents.router_agent import router_agent
from agents.customer_data_agent import customer_data_agent
from agents.support_agent import support_agent


class SupportState(TypedDict, total=False):
    messages: List[BaseMessage]
    intent: Optional[str]
    urgency: Optional[str]
    route: Optional[str]
    customer_id: Optional[int]
    response: Optional[str]
    customer_data: Optional[dict]
    tickets: Optional[list]
    customers: Optional[list]


def build_a2a_graph():
    """
    Build a LangGraph workflow that uses a messages-based A2A style state.

    Nodes:
      - RouterAgent
      - CustomerDataAgent
      - SupportAgent

    Routing:
      START -> RouterAgent
      RouterAgent -> CustomerDataAgent or SupportAgent (based on `route`)
      CustomerDataAgent -> SupportAgent
      SupportAgent -> END
    """
    graph = StateGraph(SupportState)

    graph.add_node("RouterAgent", router_agent)
    graph.add_node("CustomerDataAgent", customer_data_agent)
    graph.add_node("SupportAgent", support_agent)

    graph.add_edge(START, "RouterAgent")

    def router_next(state: Dict[str, Any]) -> str:
        route = state.get("route") or "data_then_support"
        if route == "support_only":
            return "SupportAgent"
        return "CustomerDataAgent"

    graph.add_conditional_edges(
        "RouterAgent",
        router_next,
        {
            "CustomerDataAgent": "CustomerDataAgent",
            "SupportAgent": "SupportAgent",
        },
    )

    graph.add_edge("CustomerDataAgent", "SupportAgent")
    graph.add_edge("SupportAgent", END)

    return graph.compile()
