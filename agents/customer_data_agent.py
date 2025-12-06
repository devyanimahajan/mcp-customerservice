# agents/customer_data_agent.py

from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, AIMessage

from mcp_server import (
    get_customer,
    list_customers,
    update_customer,
    create_ticket,
    get_customer_history,
)


def customer_data_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    messages: List[BaseMessage] = state.get("messages", [])
    intent: str = state.get("intent", "general_support")
    customer_id: Optional[int] = state.get("customer_id")

    customer = None
    tickets: List[Dict[str, Any]] = []
    customers: Optional[List[Dict[str, Any]]] = None

    log_parts: List[str] = []

    if customer_id is not None:
        history = get_customer_history(customer_id)
        customer = history.get("customer")
        tickets = history.get("tickets", [])
        log_parts.append(
            f"Fetched history for customer_id={customer_id} "
            f"(tickets={len(tickets)})"
        )

    if intent == "active_with_open_tickets":
        customers = list_customers(status="active", limit=100)
        log_parts.append(
            f"Listed active customers for report (count={len(customers)})"
        )

    summary = "[CustomerDataAgent] " + "; ".join(log_parts) if log_parts else "[CustomerDataAgent] No data fetch needed."
    data_message = AIMessage(content=summary)

    new_messages = messages + [data_message]

    new_state = dict(state)
    new_state.update(
        {
            "messages": new_messages,
            "customer_data": customer,
            "tickets": tickets,
            "customers": customers,
        }
    )
    return new_state
