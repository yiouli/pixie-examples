# LangChain Examples

This directory contains LangChain examples integrated with Pixie SDK.

## Examples

1. **basic_agent.py** - A simple quickstart agent that can answer questions and call tools
2. **personal_assistant.py** - Multi-agent personal assistant with subagents for calendar and email
3. **customer_support.py** - Customer support agent with state machine pattern (handoffs)
4. **sql_agent.py** - SQL database query agent (multi-turn & multi-step)
5. **langgraph_sql_agent.py** - Custom SQL agent built with LangGraph primitives
6. **langgraph_rag.py** - Custom RAG (Retrieval Augmented Generation) agent with LangGraph

## Setup

Install the required dependencies:

```bash
poetry add langchain langchain-openai langchain-community langgraph langchain-text-splitters beautifulsoup4
```

Set up environment variables in `.env`:

```bash
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
LANGSMITH_API_KEY=your_langsmith_api_key  # Optional, for tracing
LANGSMITH_TRACING=true  # Optional
```

## Running Examples

Start the Pixie server:

```bash
poetry run pixie
```

Then use GraphiQL at `http://127.0.0.1:8000/graphql` to run the agents.
