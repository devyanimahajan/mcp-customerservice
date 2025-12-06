# agents/router_agent.py

import json
import re
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def _get_last_user_message(messages: List[BaseMessage]) -> Optional[HumanMessage]:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m
    return None


def router_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Router Agent (orchestrator).

    Responsibilities:
      - Look at the latest user message in `messages`.
      - Extract a customer_id if present (e.g. "customer ID 3").
      - Classify intent + urgency (rule-based first, LLM as backup).
      - Decide route:
          "data_then_support" => CustomerDataAgent then SupportAgent
          "support_only"      => SupportAgent directly
      - Append an AIMessage describing the routing decision.
    """
    messages: List[BaseMessage] = state.get("messages", [])
    user_msg = _get_last_user_message(messages)
    query = user_msg.content if user_msg is not None else ""
    query_lower = query.lower()

    # -------- 1) Extract customer ID (regex was broken before) --------
    # IMPORTANT: single backslash \s for whitespace, not double \\s
    match = re.search(r"\b(?:customer\s+id|id)[^\d]*(\d+)\b", query_lower)
    customer_id: Optional[int] = int(match.group(1)) if match else state.get("customer_id")
    # -------- 2) Rule-based intent detection for assignment scenarios --------
    intent: Optional[str] = None

    # Scenario 3 / 4: high priority or open tickets for active customers
    if ("active customers" in query_lower and "open tickets" in query_lower) or \
       ("high priority tickets" in query_lower):
        intent = "active_with_open_tickets"

    # Scenario 5: update email and show ticket history
    elif "update my email" in query_lower and "ticket history" in query_lower:
        intent = "update_and_history"

    # Scenario 2 + escalation: cancel subscription + billing
    elif "cancel my subscription" in query_lower and "billing" in query_lower:
        intent = "billing_issue"

    # Billing-related phrases in general
    elif "charged twice" in query_lower or "refund" in query_lower or "billing issue" in query_lower:
        intent = "billing_issue"

    # Upgrade account scenario
    elif "upgrade" in query_lower or "upgrading my account" in query_lower:
        intent = "upgrade"

    # Simple lookup: “help with my account, customer ID X”
    elif "customer id" in query_lower and "help with my account" in query_lower:
        intent = "simple_lookup"

    # Fallback: ask the LLM to classify
    if intent is None:
        intent_prompt = f"""
You are a router agent for a customer support system.

User message: "{query}"

Choose an intent from:
  - "simple_lookup": get info for a single customer id
  - "upgrade": upgrade an account
  - "billing_issue": billing / charged twice / refund
  - "active_with_open_tickets": report on active customers with open tickets or high priority tickets
  - "update_and_history": update contact info and show ticket history
  - "general_support": anything else

Respond ONLY with valid JSON like:
{{
  "intent": "...",
  "urgency": "high" or "low"
}}
""".strip()
        raw = llm.invoke(intent_prompt).content
        try:
            parsed = json.loads(raw)
            intent = parsed.get("intent", "general_support")
            # we will still compute urgency below
        except Exception:
            intent = "general_support"

    # -------- 3) Urgency detection --------
    if any(phrase in query_lower for phrase in ["refund immediately", "charged twice", "urgent", "immediately"]):
        urgency = "high"
    else:
        urgency = "low"

    # -------- 4) Decide route pattern --------
    if intent in {"simple_lookup", "upgrade", "billing_issue", "update_and_history", "active_with_open_tickets"}:
        route = "data_then_support"
    else:
        route = "support_only"

    # -------- 5) Log routing decision as an AIMessage --------
    router_summary = (
        f"[RouterAgent] intent={intent}, urgency={urgency}, "
        f"customer_id={customer_id}, route={route}"
    )
    router_message = AIMessage(content=router_summary)

    new_messages = messages + [router_message]

    new_state = dict(state)
    new_state.update(
        {
            "messages": new_messages,
            "intent": intent,
            "urgency": urgency,
            "route": route,
            "customer_id": customer_id,
        }
    )
    return new_state
