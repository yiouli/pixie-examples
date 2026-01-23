"""
Basic LangChain Agent Example (Quickstart)

This example demonstrates a simple agent that can answer questions and call tools.
Based on: https://docs.langchain.com/oss/python/langchain/quickstart
"""

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from langfuse.langchain import CallbackHandler
import pixie.sdk as pixie


basic_weather_agent_prompt = pixie.create_prompt(
    "basic_weather_agent",
    description="Helpful weather assistant",
)
interactive_weather_agent_prompt = pixie.create_prompt(
    "interactive_weather_agent",
    description="Interactive weather assistant that answers questions about weather",
)

langfuse_handler = CallbackHandler()


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


@pixie.app
async def langchain_basic_weather_agent(query: str) -> str:
    """A simple weather agent that can answer questions using tools.

    Args:
        query: User's question about weather

    Returns:
        AI-generated response
    """
    # Initialize the model
    model = init_chat_model("gpt-4o-mini", temperature=0)

    # Create agent with the weather tool
    agent = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt=basic_weather_agent_prompt.compile(),
    )

    # Run the agent
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"callbacks": [langfuse_handler]},
    )

    # Return the final response
    return result["messages"][-1].content


@pixie.app
async def langchain_interactive_weather_agent() -> pixie.PixieGenerator[str, str]:
    """An interactive weather chatbot that maintains conversation.

    This agent can have multi-turn conversations with the user.

    Yields:
        AI responses to user questions
    """
    # Initialize the model
    model = init_chat_model("gpt-4o-mini", temperature=0)

    # Create agent with the weather tool
    agent = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt=interactive_weather_agent_prompt.compile(),
    )

    # Send welcome message
    yield "Hello! I'm a weather assistant. Ask me about the weather in any city!"

    # Initialize conversation history
    messages = []

    while True:
        # Get user input
        user_query = yield pixie.InputRequired(str)

        # Check for exit commands
        if user_query.lower() in {"exit", "quit", "bye", "goodbye"}:
            yield "Goodbye! Have a great day!"
            break

        # Add user message to history
        messages.append({"role": "user", "content": user_query})

        # Run agent with full conversation history
        result = agent.invoke(
            {"messages": messages}, config={"callbacks": [langfuse_handler]}
        )

        # Update history with AI response
        messages = result["messages"]

        # Yield the AI's response
        ai_response = result["messages"][-1].content
        yield ai_response
