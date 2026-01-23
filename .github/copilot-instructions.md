# Building Pixie SDK Examples - Copilot Instructions

## Overview

This guide explains how to build applications using the Pixie SDK. Pixie provides observability and control for AI agents by wrapping agent functions with the `@app` decorator.

## Core Concepts

### The `@app` Decorator

All Pixie applications must be decorated with `@app`. This decorator enables:

- Automatic discovery by the Pixie server
- Observability and tracing
- Pause/resume capabilities
- GraphQL API exposure

### Handler Function Signatures

Pixie handlers must be **async functions** or **async generators** with specific type signatures:

#### Async Function Pattern

For simple, single-turn applications:

```python
@app
async def my_app(input: InputType) -> OutputType:
    # Implementation
    return output
```

- **Input**: Can be a JSON-serializable type (`str`, `int`, `dict`, etc.) or a Pydantic `BaseModel`
- **Output**: Must be a JSON-serializable type or a Pydantic `BaseModel`

#### Async Generator Pattern

For multi-turn, interactive applications:

```python
@app
async def my_app(input: InputType) -> PixieGenerator[YieldType, SendType]:
    # Implementation
    yield output
```

- **YieldType**: Type of data yielded/sent to the client (JSON-serializable or `BaseModel`)
- **SendType**: Type of data received from user input (JSON-serializable or `BaseModel`)
- Use `PixieGenerator[YieldType, SendType]` as the return type annotation
- Import from `pixie`: `import pixie.sdk as pixie`

## Input Type Requirements

The handler function's first parameter defines what input it accepts:

### Accepting No Input

Use `None` when the application doesn't require input:

```python

@app
async def my_app(_: None) -> str:
    return "No input needed"
```

### Accepting String Input

```python
@app
async def my_app(query: str) -> str:
    return f"You asked: {query}"
```

### Accepting Structured Input with Pydantic

```python
from pydantic import BaseModel

class MyInput(BaseModel):
    topic: str
    iterations: int = 10

@app
async def my_app(config: MyInput) -> str:
    return f"Processing {config.topic} for {config.iterations} iterations"
```

## Output Type Requirements

### Simple Return Values

For async functions, return JSON-serializable data or Pydantic models:

```python
@app
async def simple_agent(query: str) -> str:
    Agent.instrument_all()
    result = await agent.run(query)
    return result.output
```

### Generator Yields

For async generators, yield values of the declared `YieldType`:

```python
@app
async def streaming_agent(query: str) -> PixieGenerator[str, None]:
    yield "Starting processing..."
    result = await agent.run(query)
    yield f"Result: {result.output}"
```

## User Input Requirement Pattern

For interactive applications that need to receive user input mid-execution, use `UserInputRequirement`:

### Basic Pattern

```python
from pixie.sdk import UserInputRequirement

@app
async def interactive_app(initial: str) -> PixieGenerator[str, str]:
    yield "Welcome! Please provide input."

    # Request string input from user
    user_input = yield UserInputRequirement(str)

    yield f"You entered: {user_input}"
```

### How It Works

1. **Yielding `UserInputRequirement`** pauses execution and requests input from the client
2. The **type parameter** specifies what type of input is expected
3. The **yielded value** is assigned to the variable (e.g., `user_input`)
4. The type parameter can be any JSON-serializable type or Pydantic `BaseModel`

### Type Parameter Examples

**String input:**

```python
user_text = yield UserInputRequirement(str)
```

**Integer input:**

```python
user_number = yield UserInputRequirement(int)
```

**Structured input with Pydantic:**

```python
from pydantic import BaseModel

class UserPreferences(BaseModel):
    language: str
    max_results: int

preferences = yield UserInputRequirement(UserPreferences)
```

**Dictionary input:**

```python
user_data = yield UserInputRequirement(dict)
```

## Complete Examples

### Example 1: Simple Single-Turn Agent

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pixie.sdk import app

# Define input model
class WeatherQuery(BaseModel):
    location: str

# Create agent
weather_agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="You are a helpful weather assistant."
)

@app
async def weather(query: WeatherQuery) -> str:
    """Simple weather query agent."""
    # Enable instrumentation for observability
    Agent.instrument_all()

    # Run agent
    result = await weather_agent.run(query.location)

    # Return output
    return result.output
```

### Example 2: Multi-Turn Chatbot with User Input

```python
from types import None
from pydantic_ai import Agent, ModelMessage, ModelRequest
from pixie.sdk import app, PixieGenerator, UserInputRequirement

chatbot = Agent(
    "openai:gpt-4o-mini",
    system_prompt="You are a friendly chatbot. Answer questions concisely."
)

@app
async def chat_with_ai(_: None) -> PixieGenerator[str, str]:
    """Interactive chatbot with conversation history."""
    # Enable instrumentation
    Agent.instrument_all()

    # Send welcome message
    yield "Hello! How can I assist you today?"

    # Maintain conversation history
    history: list[ModelMessage] = []

    while True:
        # Request user input (type: str)
        user_msg = yield UserInputRequirement(str)

        # Check for exit commands
        if user_msg.lower() in {"exit", "quit", "bye"}:
            yield "Goodbye! Have a great day!"
            break

        # Run agent with history
        ai_response = await chatbot.run(user_msg, message_history=history)

        # Update history
        history.append(ModelRequest.user_text_prompt(user_msg))
        history.append(ai_response.response)

        # Send response
        yield ai_response.output
```

### Example 3: Streaming Agent with Structured Input

```python
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pixie.sdk import app, PixieGenerator

# Define input model
class GenerationConfig(BaseModel):
    topic: str
    count: int = 5

# Create agent
generator_agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="You generate creative content on any topic."
)

@app
async def generate_content(config: GenerationConfig) -> PixieGenerator[str, None]:
    """Generate multiple outputs iteratively."""
    Agent.instrument_all()

    yield f"Starting generation on topic: {config.topic}"

    for i in range(config.count):
        result = await generator_agent.run(
            f"Generate item #{i+1} about {config.topic}"
        )
        yield f"Item {i+1}:\n{result.output}"

    yield "Generation complete!"
```

### Example 4: Agent with Tools

```python
from typing import Literal
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pixie.sdk import app

# Tool result model
class DatabaseResult(BaseModel):
    record_id: int
    data: dict

# Create agent
db_agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="You help users query a database."
)

@db_agent.tool
async def query_database(ctx: RunContext[None], query: str) -> DatabaseResult:
    """Query the database with natural language."""
    # Simulate database query
    return DatabaseResult(
        record_id=123,
        data={"result": f"Found data for: {query}"}
    )

@app
async def db_assistant(query: str) -> str:
    """Database query assistant with tools."""
    Agent.instrument_all()

    result = await db_agent.run(query)
    return result.output
```

## Best Practices

### 1. Always Enable Instrumentation

Call `Agent.instrument_all()` at the start of your handler:

```python
@app
async def my_app(input: str) -> str:
    Agent.instrument_all()  # Required for Pixie observability
    # ... rest of implementation
```

### 2. Use Descriptive Type Annotations

Clearly annotate input, output, and generator types:

```python
# Good
@app
async def chat(initial: str) -> PixieGenerator[str, str]:
    ...

# Avoid
@app
async def chat(initial):  # Missing type hints
    ...
```

### 3. Import Required Types

Import all necessary types from `pixie`:

```python
from pixie.sdk import app, PixieGenerator, UserInputRequirement
```

### 4. Handle User Input Type Safety

Match the `UserInputRequirement` type to your `PixieGenerator` send type:

```python
# YieldType=str, SendType=str
async def my_app(input: str) -> PixieGenerator[str, str]:
    user_input = yield UserInputRequirement(str)  # Matches SendType
```

### 5. Use Pydantic Models for Complex Data

Prefer Pydantic `BaseModel` for structured input/output:

```python
from pydantic import BaseModel

class Config(BaseModel):
    setting1: str
    setting2: int

class Result(BaseModel):
    status: str
    data: dict

@app
async def my_app(config: Config) -> Result:
    return Result(status="success", data={})
```

### 6. Provide Clear Docstrings

Document what your application does:

```python
@app
async def my_app(query: str) -> str:
    """Process user queries and return helpful responses.

    Args:
        query: User's natural language question

    Returns:
        AI-generated response
    """
    ...
```

## Common Patterns

### Pattern: Exit Loop on Command

```python
while True:
    user_input = yield UserInputRequirement(str)
    if user_input.lower() in {"exit", "quit", "stop"}:
        yield "Exiting..."
        break
    # Process input
```

### Pattern: Conversation History

```python
from pydantic_ai import ModelMessage, ModelRequest

history: list[ModelMessage] = []
while True:
    user_msg = yield UserInputRequirement(str)
    response = await agent.run(user_msg, message_history=history)
    history.append(ModelRequest.user_text_prompt(user_msg))
    history.append(response.response)
    yield response.output
```

### Pattern: Progress Updates

```python
yield "Starting task..."
for i in range(10):
    result = await do_work(i)
    yield f"Progress: {i+1}/10 - {result}"
yield "Task complete!"
```

### Pattern: Conditional User Input

```python
yield "Would you like more details? (yes/no)"
response = yield UserInputRequirement(str)
if response.lower() == "yes":
    yield "Here are more details..."
else:
    yield "Okay, moving on."
```

## Type Reference

### Commonly Used Types

| Type        | Usage                     | Example              |
| ----------- | ------------------------- | -------------------- |
| `str`       | String input/output       | `query: str`         |
| `int`       | Integer input/output      | `count: int`         |
| `float`     | Float input/output        | `temperature: float` |
| `bool`      | Boolean input/output      | `enabled: bool`      |
| `dict`      | Dictionary input/output   | `config: dict`       |
| `list`      | List input/output         | `items: list[str]`   |
| `None`      | No input                  | `_: None`            |
| `BaseModel` | Structured Pydantic model | `config: MyConfig`   |

### Generator Type Syntax

```python
PixieGenerator[YieldType, SendType]
```

- **YieldType**: Type of values yielded to client
- **SendType**: Type of values received via `UserInputRequirement`
- Use `None` if no user input is needed: `PixieGenerator[str, None]`

## Troubleshooting

### Issue: Handler not discovered by Pixie server

- **Solution**: Ensure function is decorated with `@app`
- **Solution**: Check that the file is in a directory scanned by Pixie (e.g., `examples/`)

### Issue: Type errors with UserInputRequirement

- **Solution**: Ensure the type passed to `UserInputRequirement(type)` matches the `SendType` in `PixieGenerator[YieldType, SendType]`

### Issue: Agent not instrumented

- **Solution**: Add `Agent.instrument_all()` at the start of your handler

### Issue: Import errors

- **Solution**: Ensure `pixie-sdk` is installed and available
- **Solution**: Check imports: `from pixie.sdk import app, PixieGenerator, UserInputRequirement`

## Testing Your Application

### Via Pixie Server

1. Start the server:

   ```bash
   poetry run pixie
   ```

2. Open GraphiQL at `http://127.0.0.1:8000/graphql`

3. Run your application via GraphQL subscription:
   ```graphql
   subscription {
     run(name: "my_app", inputData: "test input") {
       runId
       status
       data
     }
   }
   ```

### Validate Syntax

```bash
poetry run python -m py_compile examples/quickstart/my_app.py
```

### Check Imports

```bash
poetry run python -c "from examples.quickstart.my_app import my_app"
```

## Summary Checklist

When creating a Pixie application, ensure:

- [ ] Function is decorated with `@app`
- [ ] Function is `async def` (async function or async generator)
- [ ] Input parameter has proper type annotation (or `NoneType`)
- [ ] Return type is properly annotated (`str`, `BaseModel`, or `PixieGenerator[YieldType, SendType]`)
- [ ] `Agent.instrument_all()` is called at the start
- [ ] `UserInputRequirement` type matches generator's `SendType`
- [ ] All imports are correct (`from pixie.sdk import ...`)
- [ ] Function has descriptive docstring
- [ ] Code follows async/await patterns correctly
