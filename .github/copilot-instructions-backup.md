# Copilot Instructions for pixie-examples

## Repository Overview

**Purpose**: This repository contains example applications demonstrating how to use the Pixie SDK with various AI agent frameworks, primarily focused on Pydantic AI agents.

**Type**: Python examples repository  
**Size**: Small (~3 Python example files)  
**Languages/Frameworks**: Python 3.10+, Poetry, Pydantic AI, Pixie SDK  
**Target Runtime**: Python 3.12.12 (tested version)

## Project Structure

```
pixie-examples/
├── .env.example              # Environment variable template
├── .gitignore               # Git ignore file
├── README.md                # Minimal README (only contains "# pixie-examples")
├── pyproject.toml           # Poetry configuration and dependencies
├── poetry.lock              # Locked dependency versions
├── app-examples-catalog.csv # Catalog of AI agent examples from various frameworks
├── examples/                # Main examples directory
│   └── quickstart/         # Quickstart examples
│       ├── chatbot.py      # Multi-turn chatbot using Pydantic AI
│       ├── weather_agent.py # Simple weather agent with tools
│       └── sleepy_haiku_agent.py # Pause/resume demo with sleep tool
└── .vscode/
    └── launch.json         # VSCode debug configuration
```

## Key Dependencies

- **pixie-sdk**: Local development dependency at `../pixie-sdk-py` (path dependency with `develop = true`)
- **pydantic-ai**: ^1.39.0 - Primary agent framework
- **python**: >=3.10,<4.0

The project relies on a sibling directory `../pixie-sdk-py` which MUST exist for the project to function.

## Environment Setup

### Prerequisites
1. **Poetry 2.2.1+** must be installed
2. **Python 3.10-3.13** (tested with 3.12.12)
3. **Sibling pixie-sdk-py directory** must exist at `../pixie-sdk-py`

### Environment Variables
Create a `.env` file from `.env.example` with the following required keys:
```bash
OPENAI_API_KEY=<your-key>       # Required for AI model calls
LANGFUSE_PUBLIC_KEY=<...>       # Optional: For observability
LANGFUSE_SECRET_KEY=<...>       # Optional: For observability
LANGFUSE_BASE_URL=<...>         # Optional: For observability
```

**IMPORTANT**: Without `OPENAI_API_KEY`, the examples will fail at runtime when making LLM calls.

## Build & Installation

### Install Dependencies
```bash
poetry install
```

**Expected behavior**: 
- Poetry will resolve and install all dependencies from `poetry.lock`
- The local `pixie-sdk` package will be installed in development mode from `../pixie-sdk-py`
- Creates/uses a virtual environment at `.venv/` (already present in repo)

**Common warnings (can be ignored)**:
```
Warning: [tool.poetry.name] is deprecated. Use [project.name] instead.
Warning: [tool.poetry.version] is set but 'version' is not in [project.dynamic]...
Warning: [tool.poetry.description] is deprecated...
```
These are Poetry 2.x migration warnings and do not affect functionality.

### Verify Installation
```bash
poetry env info
```
Should show Python 3.12.12 and `.venv` path.

```bash
poetry show
```
Lists all installed packages (150+ dependencies including pixie-sdk, pydantic-ai, openai, etc.)

## Running Examples

### Option 1: Using Pixie Server (Recommended)
The Pixie SDK provides a GraphQL server for running agents:

```bash
poetry run pixie
```

**Expected behavior**:
- Starts server on `http://127.0.0.1:8000`
- GraphiQL interface at `http://127.0.0.1:8000/graphql`
- Auto-discovers `@pixie_app` decorated functions in the examples directory
- Watches for file changes and hot-reloads

**Common errors**:
- `[Errno 98] Address already in use`: Port 8000 is already bound. Stop the existing process or change the port.
- `No applications registered yet`: This warning is normal on startup; apps are discovered after scanning.

### Option 2: Direct Python Execution
Individual example files can be imported but may not execute standalone:

```bash
poetry run python examples/quickstart/weather_agent.py
```

**Expected behavior**: Most examples define functions decorated with `@pixie_app` and don't have `if __name__ == "__main__"` blocks, so direct execution will typically exit silently without output. These are meant to be discovered and run via the Pixie server.

## Testing

**No test infrastructure exists** in this repository. There are:
- No `pytest` configured in `pyproject.toml`
- No test files (`test_*.py` or `*_test.py`)
- No test runner scripts
- No CI/CD pipelines or GitHub Actions

To add tests, you would need to:
1. Add `pytest` to dev dependencies
2. Create a `tests/` directory
3. Configure `[tool.pytest.ini_options]` in `pyproject.toml`

## Linting & Code Quality

**No linting/formatting tools configured** in this repository:
- No `ruff`, `black`, `mypy`, `flake8`, or `pylint` in dependencies
- No configuration files (`.flake8`, `.pylintrc`, `ruff.toml`, etc.)
- No pre-commit hooks

The sibling `pixie-sdk-py` directory has `ruff` configuration, but it doesn't apply to this repo.

## Validation Workflow

Since there are no automated checks, validate changes manually:

1. **Syntax check**:
   ```bash
   poetry run python -m py_compile examples/quickstart/your_file.py
   ```

2. **Import check**:
   ```bash
   poetry run python -c "from examples.quickstart.your_file import your_function"
   ```

3. **Run via Pixie server**:
   ```bash
   poetry run pixie
   # Then test via GraphiQL at http://127.0.0.1:8000/graphql
   ```

4. **Check dependencies**:
   ```bash
   poetry check
   poetry lock --check
   ```

## Common Patterns

### Creating a New Example

1. Add a new `.py` file in `examples/quickstart/` (or create a new subdirectory under `examples/`)

2. Import required dependencies:
   ```python
   from pydantic_ai import Agent, RunContext
   from pixie import pixie_app, PixieGenerator
   from pydantic import BaseModel
   ```

3. Define your agent:
   ```python
   my_agent = Agent(
       "openai:gpt-4o-mini",  # or "openai:gpt-4o"
       system_prompt="Your system prompt here"
   )
   ```

4. Enable instrumentation in your pixie_app:
   ```python
   @pixie_app
   async def my_app(input_data: str) -> str:
       Agent.instrument_all()  # Required for Pixie observability
       result = await my_agent.run(input_data)
       return result.output
   ```

5. For agents with tools, define tool functions:
   ```python
   @my_agent.tool
   async def my_tool(ctx: RunContext[None], param: str) -> ResultModel:
       """Tool description for the LLM."""
       # Implementation
       return ResultModel(...)
   ```

6. For multi-turn interactions, use `PixieGenerator`:
   ```python
   @pixie_app
   async def chat(input: str) -> PixieGenerator[str, str]:
       Agent.instrument_all()
       yield "Initial message"
       history = []
       while True:
           user_input = yield UserInputRequirement(str)
           # Process and yield responses
   ```

## Important Notes

### Path Dependencies
- This repo has a **hard dependency** on `../pixie-sdk-py` being present
- The `pyproject.toml` specifies: `pixie-sdk = {path = "../pixie-sdk-py", develop = true}`
- If `pixie-sdk-py` is missing or moved, installation will fail
- When cloning this repo, ensure you also clone/have access to the pixie-sdk-py repository in the parent directory

### Example Catalog
- The `app-examples-catalog.csv` file contains a curated list of example implementations from major AI agent frameworks (Pydantic AI, OpenAI Agents SDK, Crew AI, Claude Agents SDK, LangChain, LangGraph, Google ADK)
- This is a reference document, not executable code
- Use it to understand what types of examples might be added to this repository

### No Main Entry Points
- Most example files don't have `if __name__ == "__main__"` blocks
- They're designed to be discovered by the Pixie server via the `@pixie_app` decorator
- Direct Python execution may produce no output

### OpenTelemetry & Observability
- Examples call `Agent.instrument_all()` to enable OpenTelemetry instrumentation
- This integrates with Langfuse (if configured) for tracing and observability
- The instrumentation is automatic and doesn't require additional setup beyond environment variables

## Commands Reference

| Action | Command | Notes |
|--------|---------|-------|
| Install | `poetry install` | Installs from lock file |
| Update deps | `poetry update` | Updates dependencies |
| Check config | `poetry check` | Validates pyproject.toml (will show deprecation warnings) |
| Show deps | `poetry show` | Lists all installed packages |
| Run example | `poetry run pixie` | Start Pixie server to run examples |
| Python version | `poetry run python --version` | Should show 3.12.12 |
| Env info | `poetry env info` | Shows virtual environment details |
| Lock file | `poetry lock` | Regenerates poetry.lock |

## Troubleshooting

### "Module 'pixie' not found"
- Ensure `poetry install` has been run
- Check that `../pixie-sdk-py` exists and is accessible
- Verify with: `poetry show pixie-sdk`

### "Address already in use" when running pixie
- Another instance of the Pixie server is running
- Find and kill: `lsof -i :8000` then `kill <PID>`
- Or use a different port (check pixie-sdk docs for port configuration)

### "No OpenAI API key"
- Create `.env` file from `.env.example`
- Add valid `OPENAI_API_KEY=sk-...`
- Restart the Pixie server

### Examples not appearing in Pixie server
- Ensure files are in `examples/` directory (or subdirectories)
- Check that functions are decorated with `@pixie_app`
- Pixie watches `/home/yiouli/repo/pixie-examples` (or wherever the repo is located)
- Check server logs for discovery messages

### Poetry warnings about deprecated fields
- These are migration warnings for Poetry 2.x
- Safe to ignore for now
- To fix, migrate `[tool.poetry]` fields to `[project]` in pyproject.toml

## Quick Start for New Contributors

1. **Clone with sibling SDK**:
   ```bash
   cd /path/to/parent
   git clone <pixie-sdk-py-repo>
   git clone <pixie-examples-repo>
   ```

2. **Setup environment**:
   ```bash
   cd pixie-examples
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

3. **Install and run**:
   ```bash
   poetry install
   poetry run pixie
   ```

4. **Access GraphiQL**: Open browser to `http://127.0.0.1:8000/graphql`

5. **Test an example** (in GraphiQL):
   ```graphql
   subscription {
     run(name: "weather", inputData: "What's the weather in San Francisco?") {
       runId
       status
       data
     }
   }
   ```

## Trust These Instructions

These instructions were generated by thoroughly exploring the repository, testing commands, and documenting actual behavior. When working on tasks:

1. **Follow these instructions first** before searching the codebase
2. **Only search** if information here is incomplete or incorrect
3. **Update these instructions** if you discover errors or new patterns
4. **Reference specific sections** when asking questions or making changes

The goal is to minimize exploration time and maximize productivity. If you find gaps, document them so the next agent benefits from your discoveries.
