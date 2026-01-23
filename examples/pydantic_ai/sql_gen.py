"""Example demonstrating SQL generation using Pydantic AI.

This shows multi-step workflow with validation and structured output.
"""

from dataclasses import dataclass
from datetime import date
from typing import Annotated, TypeAlias

from annotated_types import MinLen
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext, format_as_xml
import pixie.sdk as pixie


class SqlGenVariables(pixie.Variables):
    db_schema: str
    today_date: str
    sql_examples: str


sql_gen_agent_prompt = pixie.create_prompt(
    "sql_gen_agent",
    SqlGenVariables,
    description="Generates SQL queries from natural language for PostgreSQL databases",
)


# Simulated database schema
DB_SCHEMA = """
CREATE TABLE records (
    id bigint PRIMARY KEY,
    created_at timestamptz,
    start_timestamp timestamptz,
    end_timestamp timestamptz,
    trace_id text,
    span_id text,
    parent_span_id text,
    level log_level,
    span_name text,
    message text,
    attributes_json_schema text,
    attributes jsonb,
    tags text[],
    is_exception boolean,
    otel_status_message text,
    service_name text
);
"""

SQL_EXAMPLES = [
    {
        "request": "show me records where foobar is false",
        "response": "SELECT * FROM records WHERE attributes->>'foobar' = false",
    },
    {
        "request": 'show me records where attributes include the key "foobar"',
        "response": "SELECT * FROM records WHERE attributes ? 'foobar'",
    },
    {
        "request": "show me records from yesterday",
        "response": "SELECT * FROM records WHERE start_timestamp::date > CURRENT_TIMESTAMP - INTERVAL '1 day'",
    },
    {
        "request": 'show me error records with the tag "foobar"',
        "response": "SELECT * FROM records WHERE level = 'error' and 'foobar' = ANY(tags)",
    },
]


@dataclass
class Deps:
    """Dependencies for SQL generation (simulated database connection)."""

    validate_sql: bool = True


class Success(BaseModel):
    """Response when SQL could be successfully generated."""

    sql_query: Annotated[str, MinLen(1)]
    explanation: str = Field(
        "", description="Explanation of the SQL query, as markdown"
    )


class InvalidRequest(BaseModel):
    """Response when the user input didn't include enough information to generate SQL."""

    error_message: str


Response: TypeAlias = Success | InvalidRequest

agent = Agent[Deps, Response](
    "openai:gpt-4o-mini",
    output_type=Response,  # type: ignore
    deps_type=Deps,
)


@agent.system_prompt
async def system_prompt() -> str:
    return sql_gen_agent_prompt.compile(
        SqlGenVariables(
            db_schema=DB_SCHEMA,
            today_date=str(date.today()),
            sql_examples=format_as_xml(SQL_EXAMPLES),
        )
    )


@agent.output_validator
async def validate_output(ctx: RunContext[Deps], output: Response) -> Response:
    if isinstance(output, InvalidRequest):
        return output

    # Clean up the SQL query
    output.sql_query = output.sql_query.replace("\\", "")
    if not output.sql_query.upper().startswith("SELECT"):
        raise ModelRetry("Please create a SELECT query")

    # In a real scenario, we'd validate against an actual database
    # For now, we'll do basic validation
    if ctx.deps.validate_sql:
        # Check for dangerous operations
        upper_query = output.sql_query.upper()
        if any(
            danger in upper_query
            for danger in ["DROP", "DELETE", "UPDATE", "INSERT", "TRUNCATE"]
        ):
            raise ModelRetry("Only SELECT queries are allowed")

        # Check for required table reference
        if "FROM RECORDS" not in upper_query and 'FROM "RECORDS"' not in upper_query:
            raise ModelRetry("Query must reference the records table")

    return output


@pixie.app
async def pydantic_ai_sql_gen(query: str) -> str:
    """SQL generation agent with multi-step workflow and validation.

    This example demonstrates:
    - Dynamic system prompts with examples
    - Output validation with ModelRetry
    - Structured output with union types
    - Agent dependencies for configuration

    Args:
        query: Natural language description of desired SQL query

    Returns:
        Generated SQL query or error message
    """

    deps = Deps(validate_sql=True)
    result = await agent.run(query, deps=deps)

    if isinstance(result.output, Success):
        response = f"✅ SQL Query Generated:\n\n{result.output.sql_query}"
        if result.output.explanation:
            response += f"\n\n📝 Explanation:\n{result.output.explanation}"
        return response
    else:
        return f"❌ Error: {result.output.error_message}"


# For local testing
async def test():
    """Test function for local development."""
    test_queries = [
        "show me logs from yesterday with level 'error'",
        "find all records where foobar is true",
        "show me records from the last week",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("=" * 60)
        result = await pydantic_ai_sql_gen(query)
        print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(test())
