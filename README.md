# Customer Support Multi-Agent System (MCP + LangGraph + A2A)

This project implements a full multi-agent customer support system using:

- **MCP (Model Context Protocol)** for database-backed tools  
- **LangGraph** for agent routing and coordination  
- **A2A-style message passing** between agents  
- **SQLite** for persistent storage of customers and tickets  
- **Three specialized agents**:
  - **Router Agent**
  - **Customer Data Agent**
  - **Support Agent**

This repository fulfills all assignment requirements:
✔ Multi-agent architecture  
✔ Full MCP server implementation  
✔ A2A coordination (task allocation, negotiation, multi-step)  
✔ Jupyter notebook demonstrating all scenarios end-to-end  


---

## Project Structure
mcp_customerservice/
│
├── agents/
│ ├── router_agent.py
│ ├── customer_data_agent.py
│ ├── support_agent.py
│
├── mcp_server.py
├── database_setup.py
├── demo.ipynb
├── requirements.txt
└── README.md


---

## 1. Setup Instructions

### Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```
### Install dependencies
```bash
pip install -r requirements.txt
```

## 2. Initialise Database
Run once:
```bash
python database_setup.py
```
This creates support.db with:
- Sample customers
- Sample tickets
- Required indexes and triggers

## 3. Run MCP Server
The server implements:
- /tools/list
- /tools/call
- Full MCP-style streaming responses
Start the server
```bash
python mcp_server.py
```

## 4. Run Demo Notebook
Open jupyter lab and run demo.ipynb:
```bash
jupyter lab
demo.ipynb
```
Run all cells to see:
- Router → DataAgent → SupportAgent flow
- A2A-style communication via LangGraph messages
- Full message logs and trace output

And all required assignment scenarios:
- Simple lookup
- Upgrade flow
- Billing escalation
- Complex reporting
- Multi-intent workflow (email update + ticket history)

