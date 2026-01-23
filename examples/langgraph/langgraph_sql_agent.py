"""
LangGraph SQL Agent (Custom Implementation)

This example demonstrates building a SQL agent directly using LangGraph primitives
for deeper customization. This gives more control over the agent's behavior compared
to the higher-level LangChain agent.

Based on: https://docs.langchain.com/oss/python/langgraph/sql-agent
"""

import pathlib
import requests
from typing import Literal
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage
from langgraph.graph import START, MessagesState, StateGraph
from ..sql_utils import SQLDatabase, SQLDatabaseToolkit
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

from langfuse.langchain import CallbackHandler
import pixie.sdk as pixie


class LanggraphSqlVariables(pixie.Variables):
    dialect: str


langgraph_sql_generate_prompt = pixie.create_prompt(
    "langgraph_sql_generate_query",
    LanggraphSqlVariables,
    description="Generates SQL queries from natural language questions",
)
langgraph_sql_check_prompt = pixie.create_prompt(
    "langgraph_sql_check_query",
    LanggraphSqlVariables,
    description="Reviews and validates SQL queries for common mistakes",
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


def create_sql_graph(db: SQLDatabase, model):
    """Create a LangGraph-based SQL agent with custom workflow."""

    # Get tools from toolkit
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    tools = toolkit.get_tools()

    # Extract specific tools
    get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
    get_schema_node = ToolNode([get_schema_tool], name="get_schema")

    run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")
    run_query_node = ToolNode([run_query_tool], name="run_query")

    list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")

    # Node: List tables (forced tool call)
    def list_tables(state: MessagesState):
        tool_call = {
            "name": "sql_db_list_tables",
            "args": {},
            "id": "list_tables_call",
            "type": "tool_call",
        }
        tool_call_message = AIMessage(content="", tool_calls=[tool_call])
        tool_message = list_tables_tool.invoke(tool_call)
        response = AIMessage(f"Available tables: {tool_message.content}")
        return {"messages": [tool_call_message, tool_message, response]}

    # Node: Force model to call get_schema
    def call_get_schema(state: MessagesState):
        llm_with_tools = model.bind_tools([get_schema_tool], tool_choice="any")
        response = llm_with_tools.invoke(
            state["messages"], config={"callbacks": [langfuse_handler]}
        )
        return {"messages": [response]}

    # Node: Generate query
    def generate_query(state: MessagesState):
        generate_query_prompt_text = langgraph_sql_generate_prompt.compile(
            LanggraphSqlVariables(dialect=db.dialect)
        )
        system_message = {"role": "system", "content": generate_query_prompt_text}
        llm_with_tools = model.bind_tools([run_query_tool])
        response = llm_with_tools.invoke(
            [system_message] + state["messages"],
            config={"callbacks": [langfuse_handler]},
        )
        return {"messages": [response]}

    # Node: Check query
    def check_query(state: MessagesState):
        from langchain.messages import AIMessage as AI

        check_query_prompt_text = langgraph_sql_check_prompt.compile(
            LanggraphSqlVariables(dialect=db.dialect)
        )
        system_message = {"role": "system", "content": check_query_prompt_text}
        last_message = state["messages"][-1]
        # Only AIMessage has tool_calls
        if isinstance(last_message, AI) and last_message.tool_calls:
            tool_call = last_message.tool_calls[0]
            user_message = {"role": "user", "content": tool_call["args"]["query"]}
        else:
            # Fallback if no tool calls
            user_message = {"role": "user", "content": "Please check the query"}
        llm_with_tools = model.bind_tools([run_query_tool], tool_choice="any")
        response = llm_with_tools.invoke(
            [system_message, user_message], config={"callbacks": [langfuse_handler]}
        )
        if isinstance(last_message, AI):
            response.id = last_message.id
        return {"messages": [response]}

    # Conditional edge: Continue or end
    def should_continue(state: MessagesState) -> Literal["__end__", "check_query"]:
        from langchain.messages import AIMessage as AI

        messages = state["messages"]
        last_message = messages[-1]
        # Check if last message is AIMessage and has tool calls
        if isinstance(last_message, AI) and last_message.tool_calls:
            return "check_query"
        else:
            return "__end__"

    # Build graph
    builder = StateGraph(MessagesState)
    builder.add_node("list_tables", list_tables)
    builder.add_node("call_get_schema", call_get_schema)
    builder.add_node("get_schema", get_schema_node)
    builder.add_node("generate_query", generate_query)
    builder.add_node("check_query", check_query)
    builder.add_node("run_query", run_query_node)

    builder.add_edge(START, "list_tables")
    builder.add_edge("list_tables", "call_get_schema")
    builder.add_edge("call_get_schema", "get_schema")
    builder.add_edge("get_schema", "generate_query")
    builder.add_conditional_edges("generate_query", should_continue)
    builder.add_edge("check_query", "run_query")
    builder.add_edge("run_query", "generate_query")

    return builder.compile(checkpointer=InMemorySaver())


@pixie.app
async def langgraph_sql_agent(question: str) -> str:
    """Custom SQL agent built with LangGraph primitives.

    This agent has explicit control over the workflow:
    1. Lists all tables
    2. Gets schema for relevant tables
    3. Generates SQL query
    4. Checks query for errors
    5. Executes query
    6. Returns natural language answer

    Args:
        question: Natural language question about the database

    Returns:
        AI-generated answer based on SQL query results
    """
    # Setup database
    db = setup_database()

    # Initialize model
    model = init_chat_model("gpt-4o-mini", temperature=0)

    # Create graph
    graph = create_sql_graph(db, model)

    # Run the graph
    result = graph.invoke(
        {"messages": [{"role": "user", "content": question}]},  # type: ignore
        {
            "configurable": {"thread_id": "langgraph_sql"},
            "callbacks": [langfuse_handler],
        },
    )

    # Return the final message
    return result["messages"][-1].content
