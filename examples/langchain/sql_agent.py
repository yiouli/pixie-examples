"""
SQL Agent (Multi-Turn & Multi-Step)

This example demonstrates an agent that can answer questions about a SQL database.
The agent can:
1. Fetch available tables and schemas
2. Decide which tables are relevant
3. Generate SQL queries
4. Execute queries and handle errors
5. Formulate responses based on results

Based on: https://docs.langchain.com/oss/python/langchain/sql-agent

WARNING: Building Q&A systems of SQL databases requires executing model-generated SQL
queries. Make sure database connection permissions are scoped as narrowly as possible.
"""

import pathlib
import requests
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

from langfuse.langchain import CallbackHandler
import pixie.sdk as pixie
from ..sql_utils import SQLDatabase, SQLDatabaseToolkit


class SqlAgentVariables(pixie.Variables):
    dialect: str
    top_k: int


sql_agent_prompt = pixie.create_prompt(
    "langchain_sql_agent",
    SqlAgentVariables,
    description="SQL agent that interacts with databases and creates queries based on natural language",
)

langfuse_handler = CallbackHandler()


def setup_database():
    """Download and setup the Chinook database if not already present."""
    url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
    local_path = pathlib.Path("Chinook.db")

    if local_path.exists():
        print(f"{local_path} already exists, skipping download.")
    else:
        print("Downloading Chinook database...")
        response = requests.get(url)
        if response.status_code == 200:
            local_path.write_bytes(response.content)
            print(f"File downloaded and saved as {local_path}")
        else:
            raise Exception(
                f"Failed to download the file. Status code: {response.status_code}"
            )

    return SQLDatabase.from_uri("sqlite:///Chinook.db")


@pixie.app
async def langchain_sql_query_agent(question: str) -> str:
    """SQL database query agent that can answer questions about the Chinook database.

    The Chinook database represents a digital media store with tables for artists,
    albums, tracks, customers, invoices, etc.

    Args:
        question: Natural language question about the database

    Returns:
        AI-generated answer based on SQL query results
    """
    # Setup database
    db = setup_database()

    # Initialize model
    model = init_chat_model("gpt-4o-mini", temperature=0)

    # Create SQL toolkit with tools for database interaction
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    tools = toolkit.get_tools()

    # Format system prompt with database info
    system_prompt = sql_agent_prompt.compile(
        SqlAgentVariables(dialect=db.dialect, top_k=5)
    )

    # Create agent
    agent = create_agent(model, tools, system_prompt=system_prompt)

    # Run the agent
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"callbacks": [langfuse_handler]},
    )

    # Return the final answer
    return result["messages"][-1].content


@pixie.app
async def langchain_interactive_sql_agent() -> pixie.PixieGenerator[str, str]:
    """Interactive SQL database query agent with multi-turn conversation.

    This agent maintains conversation history and can handle follow-up questions.

    Yields:
        AI responses to database queries
    """
    # Setup database
    db = setup_database()

    # Initialize model
    model = init_chat_model("gpt-4o-mini", temperature=0)

    # Create SQL toolkit
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    tools = toolkit.get_tools()

    # Format system prompt
    system_prompt = sql_agent_prompt.compile(
        SqlAgentVariables(dialect=db.dialect, top_k=5)
    )

    # Create agent with checkpointer for conversation memory
    agent = create_agent(
        model, tools, system_prompt=system_prompt, checkpointer=InMemorySaver()
    )

    # Send welcome message
    yield f"""Welcome to the SQL Query Assistant!

I can help you query the Chinook database, which contains information about:
- Artists and Albums
- Tracks and Genres
- Customers and Invoices
- Employees and more

Available tables: {', '.join(db.get_usable_table_names())}

Ask me any question about the data!"""

    # Initialize conversation
    thread_id = "sql_thread"
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [langfuse_handler]}

    while True:
        # Get user question
        user_question = yield pixie.InputRequired(str)

        # Check for exit
        if user_question.lower() in {"exit", "quit", "bye"}:
            yield "Goodbye! Feel free to come back if you have more questions about the database."
            break

        # Process with agent
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_question}]}, config  # type: ignore
        )

        # Yield the agent's response
        yield result["messages"][-1].content
