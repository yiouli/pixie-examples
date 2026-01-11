# OpenAI Agents SDK Examples - Pixie Integration

This directory contains Pixie-integrated versions of OpenAI Agents SDK examples, demonstrating various agent patterns and workflows.

## Examples Overview

### 1. LLM as a Judge (`llm_as_a_judge.py`)

**Pattern**: Multi-Step Workflow
**Original**: [GitHub](https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/llm_as_a_judge.py)

Demonstrates the LLM-as-a-judge pattern where one agent generates content and another evaluates it iteratively.

**Features**:

- Story outline generator
- Quality evaluator agent
- Iterative improvement loop
- Feedback incorporation

**Pixie Handler**: `llm_judge_story_generator(topic: str)`

**Usage**:

```python
# Via Pixie GraphQL API
subscription {
  run(name: "llm_judge_story_generator", inputData: "a sci-fi adventure") {
    runId
    status
    data
  }
}
```

---

### 2. Customer Service (`customer_service.py`)

**Pattern**: Multi-Agent & Multi-Turn
**Original**: [GitHub](https://github.com/openai/openai-agents-python/blob/main/examples/customer_service/main.py)

A complete customer service system for an airline with multiple specialized agents that hand off to each other.

**Features**:

- Multi-agent architecture (Triage, FAQ, Seat Booking)
- Agent handoffs based on customer needs
- Shared context across agents
- Tool usage (FAQ lookup, seat updates)
- Interactive multi-turn conversation

**Pixie Handler**: `airline_customer_service()`

**Agents**:

- **Triage Agent**: Routes requests to appropriate specialists
- **FAQ Agent**: Answers common questions about baggage, seats, wifi
- **Seat Booking Agent**: Handles seat change requests

**Usage**:

```python
# Via Pixie GraphQL API
subscription {
  run(name: "airline_customer_service", inputData: null) {
    runId
    status
    data
  }
}
```

---

### 3. Routing/Handoffs (`routing.py`)

**Pattern**: Graph/State-Machine
**Original**: [GitHub](https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/routing.py)

Language-based routing where a triage agent hands off to specialized language agents.

**Features**:

- Automatic language detection
- Agent routing based on language
- Streaming responses
- Support for French, Spanish, and English

**Pixie Handlers**:

- `multilingual_routing()` - No initial message required
- `multilingual_routing_simple(initial_message: str)` - Start with a message

**Usage**:

```python
# Interactive version
subscription {
  run(name: "multilingual_routing", inputData: null) {
    runId
    status
    data
  }
}

# With initial message
subscription {
  run(name: "multilingual_routing_simple", inputData: "Bonjour, comment allez-vous?") {
    runId
    status
    data
  }
}
```

---

### 4. Financial Research Agent (`financial_research_agent.py`)

**Pattern**: Multi-Step Workflow with Multi-Agent
**Original**: [GitHub](https://github.com/openai/openai-agents-python/tree/main/examples/financial_research_agent)

Comprehensive financial research workflow with specialized agents for analysis.

**Features**:

- Structured research workflow
- Parallel web searches
- Specialist analysts (fundamentals, risk)
- Report synthesis
- Quality verification
- Markdown report generation

**Architecture**:

1. **Planner Agent** - Creates search strategy
2. **Search Agent** - Performs web searches
3. **Fundamentals Analyst** - Analyzes financial metrics
4. **Risk Analyst** - Identifies risk factors
5. **Writer Agent** - Synthesizes final report
6. **Verifier Agent** - Quality checks

**Pixie Handler**: `financial_research(query: str)`

**Usage**:

```python
# Via Pixie GraphQL API
subscription {
  run(
    name: "financial_research",
    inputData: "Write up an analysis of Apple Inc.'s most recent quarter."
  ) {
    runId
    status
    data
  }
}
```

**Example Queries**:

- "Analyze Tesla's financial performance and growth prospects"
- "What are the key risks facing Microsoft in 2026?"
- "Provide a comprehensive analysis of NVIDIA's market position"

---

## Setup

### 1. Install Dependencies

The OpenAI Agents SDK is already installed in the project:

```bash
poetry install
```

### 2. Configure Environment Variables

Make sure you have your OpenAI API key configured in `.env`:

```bash
OPENAI_API_KEY=sk-...
```

### 3. Run Pixie Server

Start the Pixie server to access the examples:

```bash
poetry run pixie
```

Then navigate to `http://127.0.0.1:8000/graphql` to use the GraphiQL interface.

---

## Key Patterns Demonstrated

### Multi-Turn Conversations

Examples use `PixieGenerator[str, str]` with `UserInputRequirement(str)` for interactive dialogues:

```python
@app
async def my_chatbot(_: None) -> PixieGenerator[str, str]:
    yield "Hello! How can I help?"

    while True:
        user_input = yield UserInputRequirement(str)
        # Process and respond
```

### Agent Handoffs

Multiple agents collaborate by handing off to specialists:

```python
triage_agent = Agent(
    handoffs=[specialist_agent_1, specialist_agent_2]
)
```

### Streaming Responses

Real-time response streaming for better UX:

```python
result = Runner.run_streamed(agent, inputs)
async for event in result.stream_events():
    if isinstance(data, ResponseTextDeltaEvent):
        yield data.delta
```

### Structured Outputs

Using Pydantic models for type-safe outputs:

```python
class ReportData(BaseModel):
    summary: str
    full_report: str

agent = Agent(output_type=ReportData)
```

---

## Differences from Original Examples

These Pixie-integrated versions differ from the originals in several ways:

1. **Async Generators**: Use `PixieGenerator` for streaming outputs
2. **User Input**: Use `UserInputRequirement` instead of `input()`
3. **Instrumentation**: Pixie automatically handles tracing
4. **No Main Loop**: No `if __name__ == "__main__"` - runs via Pixie server
5. **GraphQL Interface**: Accessed via GraphQL subscriptions instead of CLI

---

## Testing

Check for syntax errors:

```bash
poetry run python -m py_compile examples/openai_agents_sdk/*.py
```

Verify imports:

```bash
poetry run python -c "from examples.openai_agents_sdk.llm_as_a_judge import llm_judge_story_generator"
poetry run python -c "from examples.openai_agents_sdk.customer_service import airline_customer_service"
poetry run python -c "from examples.openai_agents_sdk.routing import multilingual_routing"
poetry run python -c "from examples.openai_agents_sdk.financial_research_agent import financial_research"
```

---

## Resources

- [OpenAI Agents SDK Documentation](https://github.com/openai/openai-agents-python)
- [Pixie SDK Documentation](../.github/copilot-instructions.md)
- [Original Examples](https://github.com/openai/openai-agents-python/tree/main/examples)

---

## Contributing

To add more OpenAI Agents SDK examples:

1. Create a new `.py` file in this directory
2. Import required modules from `agents` and `pixie`
3. Define your agents and tools
4. Create a Pixie handler with `@app` decorator
5. Use `PixieGenerator` for multi-turn or streaming applications
6. Update this README with example details
