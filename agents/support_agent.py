# agents/support_agent.py

from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from mcp_server import (
    get_customer_history,
    create_ticket,
    update_customer,
)


def _get_last_user_message(messages: List[BaseMessage]) -> Optional[HumanMessage]:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m
    return None


def support_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Support Agent.

    Uses:
      - intent, urgency, customer_id decided by RouterAgent
      - customer_data, tickets, customers set by CustomerDataAgent

    Responsibilities:
      - Simple lookup of customer info
      - Upgrade flows
      - Billing issues / escalation (creating tickets)
      - Reports on active customers with high-priority tickets
      - Multi-intent: update email + show ticket history
      - Fallback general support
    """
    messages: List[BaseMessage] = state.get("messages", [])
    intent: str = state.get("intent", "general_support")
    urgency: str = state.get("urgency", "low")
    customer_id: Optional[int] = state.get("customer_id")

    customer = state.get("customer_data")
    tickets: List[Dict[str, Any]] = state.get("tickets") or []
    customers: Optional[List[Dict[str, Any]]] = state.get("customers")

    # Get the raw user query text
    user_msg = _get_last_user_message(messages)
    query = user_msg.content if user_msg is not None else state.get("input", "")

    support_log = ""

    # -------------------------------------------------
    # Scenario 1: Simple Query - "Get customer information for ID 5"
    # -------------------------------------------------
    if intent == "simple_lookup":
        if not customer:
            response_text = "I could not find that customer. Please double check the ID."
            support_log = "[SupportAgent] simple_lookup but customer not found."
        else:
            response_text = (
                f"Customer {customer['id']}: {customer['name']}\n"
                f"Email: {customer.get('email')}\n"
                f"Phone: {customer.get('phone')}\n"
                f"Status: {customer.get('status')}"
            )
            support_log = "[SupportAgent] Handled simple_lookup."
        support_message = AIMessage(content=support_log)
        new_messages = messages + [support_message]
        new_state = dict(state)
        new_state.update({"messages": new_messages, "response": response_text})
        return new_state

    # -------------------------------------------------
    # Scenario 2: Coordinated Query (Upgrade)
    # "I am customer ID 3 and need help upgrading my account"
    # -------------------------------------------------
    if intent == "upgrade":
        if not customer:
            response_text = (
                "I could not find your account. Please provide a valid customer ID."
            )
            support_log = "[SupportAgent] upgrade intent but customer not found."
        else:
            response_text = (
                f"Hi {customer['name']}, I can help you upgrade your account. "
                f"I will use the email on file ({customer.get('email')}) to send a confirmation."
            )
            support_log = "[SupportAgent] Handled upgrade flow."
        support_message = AIMessage(content=support_log)
        new_messages = messages + [support_message]
        new_state = dict(state)
        new_state.update({"messages": new_messages, "response": response_text})
        return new_state

    # -------------------------------------------------
    # Scenario 4: Escalation (Refund, urgent billing issue)
    # "I am customer ID 2 and have been charged twice, please refund immediately!"
    # -------------------------------------------------
    if intent == "billing_issue":
        if customer_id is None:
            response_text = (
                "I am sorry about the billing issue. Please provide your customer ID "
                "so that I can create a ticket and investigate."
            )
            support_log = "[SupportAgent] Billing issue but no customer_id provided."
        else:
            priority = "high" if urgency == "high" else "medium"
            created = create_ticket(customer_id, query, priority=priority)
            response_text = (
                "I am sorry about the billing issue. I have created a ticket "
                f"#{created['id']} with priority {priority}. "
                "Our billing team will review your charge and process any refund that is due."
            )
            support_log = (
                f"[SupportAgent] Created billing ticket #{created['id']} "
                f"for customer_id={customer_id} with priority={priority}."
            )
        support_message = AIMessage(content=support_log)
        new_messages = messages + [support_message]
        new_state = dict(state)
        new_state.update({"messages": new_messages, "response": response_text})
        return new_state

    # -------------------------------------------------
    # Scenario 3: Complex Query
    # "Show me all active customers who have open tickets"
    # (implemented here as high-priority ticket status for active customers)
    # -------------------------------------------------
    if intent == "active_with_open_tickets" and customers is not None:
        lines: List[str] = []
        for c in customers:
            history = get_customer_history(c["id"])
            high_priority_open = [
                t
                for t in history.get("tickets", [])
                if t["status"] == "open" and t["priority"] == "high"
            ]
            if high_priority_open:
                lines.append(
                    f"- Customer {c['id']} ({c['name']}): "
                    f"{len(high_priority_open)} high-priority ticket(s)"
                )

        if not lines:
            response_text = "There are no active customers with open high-priority tickets right now."
        else:
            response_text = (
                "High priority ticket status for active customers:\n" + "\n".join(lines)
            )

        support_log = "[SupportAgent] Generated report for active customers with high priority tickets."
        support_message = AIMessage(content=support_log)
        new_messages = messages + [support_message]
        new_state = dict(state)
        new_state.update({"messages": new_messages, "response": response_text})
        return new_state

    # -------------------------------------------------
    # Scenario 5: Multi-Intent
    # "Update my email to new@email.com and show my ticket history"
    # -------------------------------------------------
    if intent == "update_and_history" and customer_id is not None:
        # Robust email regex over the full query
        import re

        email_pattern = r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
        match = re.search(email_pattern, query)

        if match:
            new_email = match.group(0)

            # Update customer record
            updated = update_customer(customer_id, {"email": new_email})

            # Refresh history after update
            history = get_customer_history(customer_id)
            ticket_list = history.get("tickets", [])
            ticket_lines = [
                f"- [{t['status']}] {t['issue']}"
                for t in ticket_list
            ]

            response_text = (
                f"Updated email to {updated.get('email')} for customer {updated['name']}.\n\n"
                f"Ticket history ({len(ticket_list)} total):\n"
                + ("\n".join(ticket_lines) if ticket_lines else "No tickets found.")
            )
            support_log = (
                f"[SupportAgent] Updated email to {new_email} and summarized ticket history."
            )
        else:
            response_text = "I could not find a valid email address to update."
            support_log = "[SupportAgent] update_and_history but no valid email found."

        support_message = AIMessage(content=support_log)
        new_messages = messages + [support_message]
        new_state = dict(state)
        new_state.update({"messages": new_messages, "response": response_text})
        return new_state

    # -------------------------------------------------
    # Fallback: General support
    # -------------------------------------------------
    response_text = "I have logged your issue and will route it to the appropriate team."
    support_log = "[SupportAgent] Fallback general_support."
    support_message = AIMessage(content=support_log)
    new_messages = messages + [support_message]
    new_state = dict(state)
    new_state.update({"messages": new_messages, "response": response_text})
    return new_state
