"""
MCP Server for Customer Support Database.

Implements:
  - tools/list
  - tools/call

Backed by SQLite database "support.db" created by database_setup.py.
"""

from typing import Any, Dict, List, Callable, Optional

import json
import sqlite3

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

DB_PATH = "support.db"

# =====================================================
# Database helper functions (MCP tool implementations)
# =====================================================

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_customer(customer_id: int) -> Dict[str, Any]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM customers WHERE id = ?", (customer_id,))
        row = cur.fetchone()
        return dict(row) if row else {}
    finally:
        conn.close()


def list_customers(status: str = "active", limit: int = 20) -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM customers WHERE status = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (status, limit),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def update_customer(customer_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
    allowed_fields = {"name", "email", "phone", "status"}
    fields = [k for k in data.keys() if k in allowed_fields]
    if not fields:
        raise ValueError("No valid fields to update.")

    set_clause = ", ".join(f"{f} = ?" for f in fields)
    values = [data[f] for f in fields]

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            f"UPDATE customers SET {set_clause}, updated_at = CURRENT_TIMESTAMP "
            "WHERE id = ?",
            (*values, customer_id),
        )
        conn.commit()
        cur.execute("SELECT * FROM customers WHERE id = ?", (customer_id,))
        row = cur.fetchone()
        return dict(row) if row else {}
    finally:
        conn.close()


def create_ticket(customer_id: int, issue: str, priority: str = "medium") -> Dict[str, Any]:
    if priority not in {"low", "medium", "high"}:
        raise ValueError("priority must be one of: low, medium, high")

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO tickets (customer_id, issue, status, priority, created_at)
            VALUES (?, ?, 'open', ?, CURRENT_TIMESTAMP)
            """,
            (customer_id, issue, priority),
        )
        conn.commit()
        ticket_id = cur.lastrowid
        cur.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,))
        row = cur.fetchone()
        return dict(row)
    finally:
        conn.close()


def get_customer_history(customer_id: int) -> Dict[str, Any]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM customers WHERE id = ?", (customer_id,))
        cust = cur.fetchone()
        if not cust:
            return {"customer": None, "tickets": []}

        cur.execute(
            "SELECT * FROM tickets WHERE customer_id = ? "
            "ORDER BY created_at DESC",
            (customer_id,),
        )
        tickets = [dict(r) for r in cur.fetchall()]
        return {"customer": dict(cust), "tickets": tickets}
    finally:
        conn.close()


# =====================================================
# MCP tool registry and schemas
# =====================================================

class ToolDef:
    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
    ):
        self.name = name
        self.description = description
        self.func = func
        self.input_schema = input_schema
        self.output_schema = output_schema


TOOLS: Dict[str, ToolDef] = {
    "get_customer": ToolDef(
        name="get_customer",
        description="Get a single customer by id.",
        func=get_customer,
        input_schema={
            "type": "object",
            "properties": {
                "customer_id": {"type": "integer"},
            },
            "required": ["customer_id"],
        },
        output_schema={"type": "object"},
    ),
    "list_customers": ToolDef(
        name="list_customers",
        description="List customers by status with an optional limit.",
        func=list_customers,
        input_schema={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["active", "disabled"],
                    "default": "active",
                },
                "limit": {"type": "integer", "default": 20},
            },
            "required": [],
        },
        output_schema={"type": "array", "items": {"type": "object"}},
    ),
    "update_customer": ToolDef(
        name="update_customer",
        description="Update fields of a customer record.",
        func=update_customer,
        input_schema={
            "type": "object",
            "properties": {
                "customer_id": {"type": "integer"},
                "data": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                        "phone": {"type": "string"},
                        "status": {"type": "string", "enum": ["active", "disabled"]},
                    },
                    "additionalProperties": False,
                },
            },
            "required": ["customer_id", "data"],
        },
        output_schema={"type": "object"},
    ),
    "create_ticket": ToolDef(
        name="create_ticket",
        description="Create a new support ticket for a customer.",
        func=create_ticket,
        input_schema={
            "type": "object",
            "properties": {
                "customer_id": {"type": "integer"},
                "issue": {"type": "string"},
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "default": "medium",
                },
            },
            "required": ["customer_id", "issue"],
        },
        output_schema={"type": "object"},
    ),
    "get_customer_history": ToolDef(
        name="get_customer_history",
        description="Get a customer and all of their tickets.",
        func=get_customer_history,
        input_schema={
            "type": "object",
            "properties": {
                "customer_id": {"type": "integer"},
            },
            "required": ["customer_id"],
        },
        output_schema={"type": "object"},
    ),
}


# =====================================================
# FastAPI app implementing MCP-like tools/list and tools/call
# =====================================================

app = FastAPI(title="Customer Support MCP Server")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/tools/list")
async def tools_list():
    """
    MCP-style tools/list endpoint.

    Returns metadata that an MCP client (or MCP Inspector) can use
    to discover available tools and their input schemas.
    """
    tools_payload = []
    for tool in TOOLS.values():
        tools_payload.append(
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
                "output_schema": tool.output_schema,
            }
        )
    return {"tools": tools_payload}


@app.post("/tools/call")
async def tools_call(body: Dict[str, Any]):
    """
    MCP-style tools/call endpoint.

    Expected JSON body:
      {
        "name": "<tool_name>",
        "arguments": { ... }
      }

    Response is streamed as newline-delimited JSON chunks to
    demonstrate a "streamable" HTTP protocol.
    """
    name = body.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="Missing 'name' in request body.")

    tool_def = TOOLS.get(name)
    if not tool_def:
        raise HTTPException(status_code=404, detail=f"Unknown tool '{name}'.")

    arguments = body.get("arguments", {}) or {}

    async def event_stream():
        # Start event
        start_msg = {"event": "start", "tool": name}
        yield json.dumps(start_msg) + "\n"

        try:
            result = tool_def.func(**arguments)
            result_msg = {
                "event": "result",
                "tool": name,
                "output": result,
            }
            yield json.dumps(result_msg) + "\n"
        except Exception as e:
            error_msg = {
                "event": "error",
                "tool": name,
                "error": str(e),
            }
            yield json.dumps(error_msg) + "\n"

        # End event
        end_msg = {"event": "end", "tool": name}
        yield json.dumps(end_msg) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/json",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("mcp_server:app", host="0.0.0.0", port=8000, reload=True)
