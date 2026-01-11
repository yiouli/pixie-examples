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
import pixie
from ..sql_utils import SQLDatabase, SQLDatabaseToolkit


# System prompt for SQL agent
SQL_AGENT_PROMPT = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
"""


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
    system_prompt = SQL_AGENT_PROMPT.format(dialect=db.dialect, top_k=5)

    # Create agent
    agent = create_agent(model, tools, system_prompt=system_prompt)

    # Run the agent
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

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
    system_prompt = SQL_AGENT_PROMPT.format(dialect=db.dialect, top_k=5)

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
    config = {"configurable": {"thread_id": thread_id}}

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
