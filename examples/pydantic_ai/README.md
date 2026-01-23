# PydanticAI Examples with Pixie SDK

This directory contains examples from the [PydanticAI Catalog](https://github.com/pydantic/pydantic-ai) integrated with Pixie SDK for observability and execution.

## Examples from Catalog

All examples are sourced from the official PydanticAI catalog and adapted for Pixie SDK.

### 1. pydantic_model.py - Pydantic Model (Quickstart)

Simple example demonstrating structured output extraction from text.

**Pixie Handler:** `pydantic_model_example`
**Documentation:** [pydantic-model](https://ai.pydantic.dev/examples/pydantic-model/)
**Source:** [pydantic_model.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/pydantic_model.py)

- **Pattern:** Async function
- **Input:** String query (e.g., "The windy city in the US of A.")
- **Output:** Structured `MyModel` with city and country
- **Features:**
  - Basic agent setup with structured output
  - Pydantic model validation
  - Simple query → structured response

### 2. bank_support.py - Bank Support (Multi-Turn Chatbot)

Interactive bank support agent with tools and conversation history.

**Pixie Handler:** `bank_support_agent`
**Documentation:** [bank-support](https://ai.pydantic.dev/examples/bank-support/)
**Source:** [bank_support.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/bank_support.py)

- **Pattern:** Async generator (multi-turn)
- **Input:** None (interactive)
- **Output:** Generator[str, str] for conversation
- **Features:**
  - Multi-turn conversation flow
  - Tool usage (customer_balance)
  - Dynamic system instructions
  - SQLite database integration
  - Interactive user input with `UserInputRequirement`

### 3. flight_booking.py - Flight Booking (Multi-Agent)

Multi-agent system for flight search, extraction, and seat selection.

**Pixie Handler:** `flight_booking_example`
**Documentation:** [flight-booking](https://ai.pydantic.dev/examples/flight-booking/)
**Source:** [flight_booking.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/flight_booking.py)

- **Pattern:** Async generator (interactive)
- **Input:** None (interactive)
- **Output:** Generator[str, str] for booking flow
- **Features:**
  - **Agent Delegation:** search_agent delegates to extraction_agent
  - **Programmatic Hand-off:** flight search → seat selection
  - Multi-agent collaboration
  - Usage limits and tracking
  - Output validation with constraints
  - Interactive confirmation workflow

### 4. question_graph.py - Question Graph (Graph/State-Machine)

Graph-based Q&A system using pydantic_graph for state machine control flow.

**Pixie Handler:** `question_graph_example`
**Documentation:** [question-graph](https://ai.pydantic.dev/examples/question-graph/)
**Source:** [question_graph.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/question_graph.py)

- **Pattern:** Async generator (interactive with graph)
- **Input:** None (interactive)
- **Output:** Generator[str, str] for Q&A flow
- **Features:**
  - **pydantic_graph** for state machine control flow
  - Graph nodes: Ask → Answer → Evaluate → Reprimand/End
  - State management across nodes
  - Conditional branching (correct/incorrect)
  - Message history per agent

**Graph Flow:**

```
Ask (generate question)
  ↓
Answer (get user input)
  ↓
Evaluate (check answer)
  ↓
  ├─ Correct → End (success)
  └─ Wrong → Reprimand → Ask (retry)
```

### 5. sql_gen.py - SQL Generation (Multi-Step Workflow)

SQL query generation with validation and structured output.

**Pixie Handler:** `sql_gen_example`
**Documentation:** [sql-gen](https://ai.pydantic.dev/examples/sql-gen/)
**Source:** [sql_gen.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/sql_gen.py)

- **Pattern:** Async function
- **Input:** String query (natural language)
- **Output:** String (SQL query or error message)
- **Features:**
  - Dynamic system prompts with SQL examples
  - Output validation with `ModelRetry`
  - Structured output with union types (`Success | InvalidRequest`)
  - Multi-step validation workflow
  - SQL safety checks (prevents dangerous operations)
  - Schema-aware query generation

## Setup

### 1. Install Dependencies

From the project root:

```bash
poetry install
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and add your API keys:

```bash
cp examples/pydantic_ai/.env.example examples/pydantic_ai/.env
```

Edit `.env` and add:

```bash
# Required for most examples
OPENAI_API_KEY=sk-...

# Optional
LOGFIRE_TOKEN=...  # For observability
```

### 3. Start Pixie Server

```bash
poetry run pixie
```

The server will start at http://127.0.0.1:8000

## Running Examples

### Via GraphQL Playground

Open http://127.0.0.1:8000/graphql and run subscriptions:

#### Example 1: Pydantic Model

```graphql
subscription {
  run(name: "pydantic_model_example", inputData: "The capital of France") {
    runId
    status
    data
  }
}
```

#### Example 2: Bank Support (Interactive)

```graphql
subscription {
  run(name: "bank_support_agent") {
    runId
    status
    data
  }
}
```

#### Example 3: Flight Booking

```graphql
subscription {
  run(name: "flight_booking_example") {
    runId
    status
    data
  }
}
```

#### Example 4: Question Graph

```graphql
subscription {
  run(name: "question_graph_example") {
    runId
    status
    data
  }
}
```

#### Example 5: SQL Generation

```graphql
subscription {
  run(name: "sql_gen_example", inputData: "show me error logs from yesterday") {
    runId
    status
    data
  }
}
```

## Architecture Patterns

### Async Functions vs Generators

- **Async Function** (`async def → Type`): Single request-response (pydantic_model, sql_gen)
- **Async Generator** (`async def → PixieGenerator[YieldType, SendType]`): Multi-turn (bank_support, flight_booking, question_graph)

### Agent Patterns

1. **Single Agent** - One agent handles workflow
2. **Multi-Turn Chat** - Agent + user interaction loop
3. **Agent Delegation** - One agent calls another
4. **Programmatic Hand-off** - Sequential agent execution
5. **Graph-Based Control** - pydantic_graph state machines

## Integration with Pixie SDK

All examples use:

```python
from pixie.sdk import app, PixieGenerator, UserInputRequirement

@app
async def my_handler(input: InputType) -> OutputType:
    Agent.instrument_all()  # Enable observability
    # ... implementation
```

## Resources

- [PydanticAI Documentation](https://ai.pydantic.dev/)
- [Pixie SDK Guide](../../.github/copilot-instructions.md)
- [PydanticAI GitHub](https://github.com/pydantic/pydantic-ai)
