"""Bank support agent example using Pydantic AI.

This example demonstrates:
- Dynamic system prompts with agent instructions
- Structured output types
- Tools for querying dependencies
- Integration with Pixie SDK

Run with:
    poetry run pixie

Then query via GraphQL:
    subscription {
      run(name: "bank_support_agent", inputData: "What is my balance?") {
        runId
        status
        data
      }
    }
"""

import sqlite3
from dataclasses import dataclass
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
import pixie.sdk as pixie


bank_support_agent_prompt = pixie.create_prompt(
    "bank_support_agent",
    description="Support agent that helps customers and assesses risk levels",
)


@dataclass
class DatabaseConn:
    """A wrapper over the SQLite connection."""

    sqlite_conn: sqlite3.Connection

    async def customer_name(self, *, customer_id: int) -> str | None:
        res = self.sqlite_conn.execute(
            "SELECT name FROM customers WHERE id=?", (customer_id,)
        )
        row = res.fetchone()
        if row:
            return row[0]
        return None

    async def customer_balance(self, *, customer_id: int) -> float:
        res = self.sqlite_conn.execute(
            "SELECT balance FROM customers WHERE id=?", (customer_id,)
        )
        row = res.fetchone()
        if row:
            return row[0]
        else:
            raise ValueError("Customer not found")


@dataclass
class SupportDependencies:
    customer_id: int
    db: DatabaseConn


class SupportOutput(BaseModel):
    """Output model for bank support agent."""

    support_advice: str
    """Advice returned to the customer"""
    block_card: bool
    """Whether to block their card or not"""
    risk: int
    """Risk level of query"""


@pixie.app
async def pydantic_ai_bank_support_agent() -> pixie.PixieGenerator[str, str]:
    """Interactive bank support agent.

    This agent helps customers with banking queries, can check balances,
    and assesses risk levels of queries.

    Yields:
        str: Agent responses and support advice

    Receives:
        str: User queries via InputRequired
    """

    # Create the support agent
    support_agent = Agent(
        "openai:gpt-4o-mini",
        deps_type=SupportDependencies,
        output_type=SupportOutput,
        instructions=bank_support_agent_prompt.compile(),
    )

    @support_agent.instructions
    async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
        customer_name = await ctx.deps.db.customer_name(
            customer_id=ctx.deps.customer_id
        )
        return f"The customer's name is {customer_name!r}"

    @support_agent.tool
    async def customer_balance_tool(ctx: RunContext[SupportDependencies]) -> str:
        """Returns the customer's current account balance."""
        balance = await ctx.deps.db.customer_balance(
            customer_id=ctx.deps.customer_id,
        )
        return f"${balance:.2f}"

    # Initialize database connection (in-memory for demo)
    with sqlite3.connect(":memory:") as con:
        cur = con.cursor()
        cur.execute("CREATE TABLE customers(id, name, balance)")
        cur.execute(
            """
            INSERT INTO customers VALUES
                (123, 'John', 123.45),
                (456, 'Jane', 567.89)
        """
        )
        con.commit()

        # Default customer
        customer_id = 123
        deps = SupportDependencies(
            customer_id=customer_id, db=DatabaseConn(sqlite_conn=con)
        )

        yield "Welcome to Bank Support! I'm here to help you with your account."

        while True:
            # Get user query
            user_query = yield pixie.InputRequired(str)

            # Check for exit commands
            if user_query.lower() in {"exit", "quit", "bye"}:
                yield "Thank you for using our support service. Have a great day!"
                break

            # Run the agent
            result = await support_agent.run(user_query, deps=deps)

            # Format the response
            output = result.output
            response = f"""
Support Advice: {output.support_advice}
Card Status: {"🔒 BLOCKED" if output.block_card else "✓ Active"}
Risk Level: {output.risk}/10
"""
            yield response.strip()
