"""
Routing/Handoffs Pattern - Pixie Integration

This example demonstrates agent routing where a triage agent hands off to specialized
language agents based on the user's language preference.

Pattern: Graph/State-Machine (Routing)
Original: https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/routing.py
"""

from __future__ import annotations

from types import NoneType

from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from agents import Agent, RawResponsesStreamEvent, Runner, TResponseInputItem
from pixie import pixie_app, PixieGenerator, UserInputRequirement


# ============================================================================
# LANGUAGE AGENTS
# ============================================================================

french_agent = Agent(
    name="french_agent",
    instructions="You only speak French",
)

spanish_agent = Agent(
    name="spanish_agent",
    instructions="You only speak Spanish",
)

english_agent = Agent(
    name="english_agent",
    instructions="You only speak English",
)

triage_agent = Agent(
    name="triage_agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[french_agent, spanish_agent, english_agent],
)


@pixie_app
async def multilingual_routing(_: NoneType) -> PixieGenerator[str, str]:
    """
    Multi-agent language routing system with streaming responses.

    The triage agent receives messages and hands off to the appropriate language
    specialist agent (French, Spanish, or English) based on the detected language.

    Responses are streamed in real-time to provide immediate feedback.

    Yields:
        Streamed agent responses

    Receives:
        User messages in any supported language via UserInputRequirement
    """
    agent = triage_agent
    inputs: list[TResponseInputItem] = []

    yield "Hi! We speak French, Spanish and English. How can I help?"
    yield "(Type 'exit' to quit)"

    while True:
        # Get user message
        user_msg = yield UserInputRequirement(str)

        # Check for exit
        if user_msg.lower() in {"exit", "quit", "bye"}:
            yield "Goodbye! Au revoir! ¡Adiós!"
            break

        inputs.append({"content": user_msg, "role": "user"})

        # Run agent with streaming
        result = Runner.run_streamed(
            agent,
            input=inputs,
        )

        # Stream the response
        current_response = []
        async for event in result.stream_events():
            if not isinstance(event, RawResponsesStreamEvent):
                continue

            data = event.data

            if isinstance(data, ResponseTextDeltaEvent):
                # Stream text deltas
                current_response.append(data.delta)
                yield data.delta

            elif isinstance(data, ResponseContentPartDoneEvent):
                # Content part complete
                yield "\n"

        # Update state for next turn
        inputs = result.to_input_list()
        agent = result.current_agent

        # Show which agent handled the request
        agent_name = agent.name
        yield f"\n[Handled by: {agent_name}]"


@pixie_app
async def multilingual_routing_simple(initial_message: str) -> PixieGenerator[str, str]:
    """
    Simplified multilingual routing with single initial message.

    This version accepts an initial message and then enters an interactive loop.
    Good for testing with a specific language right away.

    Args:
        initial_message: The first message to send (in any supported language)

    Yields:
        Streamed agent responses

    Receives:
        Follow-up user messages via UserInputRequirement
    """
    agent = triage_agent
    inputs: list[TResponseInputItem] = [{"content": initial_message, "role": "user"}]

    yield f"Processing your message: {initial_message[:50]}...\n"

    # Process initial message
    result = Runner.run_streamed(agent, input=inputs)

    async for event in result.stream_events():
        if not isinstance(event, RawResponsesStreamEvent):
            continue

        data = event.data
        if isinstance(data, ResponseTextDeltaEvent):
            yield data.delta
        elif isinstance(data, ResponseContentPartDoneEvent):
            yield "\n"

    inputs = result.to_input_list()
    agent = result.current_agent
    yield f"\n[Agent: {agent.name}]"

    # Continue conversation
    yield "\nContinue the conversation (type 'exit' to quit):"

    while True:
        user_msg = yield UserInputRequirement(str)

        if user_msg.lower() in {"exit", "quit", "bye"}:
            yield "Session ended. Merci! ¡Gracias! Thank you!"
            break

        inputs.append({"content": user_msg, "role": "user"})

        result = Runner.run_streamed(agent, input=inputs)

        async for event in result.stream_events():
            if not isinstance(event, RawResponsesStreamEvent):
                continue

            data = event.data
            if isinstance(data, ResponseTextDeltaEvent):
                yield data.delta
            elif isinstance(data, ResponseContentPartDoneEvent):
                yield "\n"

        inputs = result.to_input_list()
        agent = result.current_agent
        yield f"\n[Agent: {agent.name}]"
