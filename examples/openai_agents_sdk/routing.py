"""
Routing/Handoffs Pattern - Pixie Integration

This example demonstrates agent routing where a triage agent hands off to specialized
language agents based on the user's language preference.

Pattern: Graph/State-Machine (Routing)
Original: https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/routing.py
"""

from __future__ import annotations


from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from agents import Agent, RawResponsesStreamEvent, Runner, TResponseInputItem
import pixie.sdk as pixie


french_agent_prompt = pixie.create_prompt(
    "french_agent",
    description="Agent that only speaks French",
)
spanish_agent_prompt = pixie.create_prompt(
    "spanish_agent",
    description="Agent that only speaks Spanish",
)
english_agent_prompt = pixie.create_prompt(
    "english_agent",
    description="Agent that only speaks English",
)
triage_routing_agent_prompt = pixie.create_prompt(
    "triage_routing_agent",
    description="Routes requests to appropriate language-specific agent",
)

# ============================================================================
# LANGUAGE AGENTS
# ============================================================================

# Agents will be lazily initialized
_french_agent: Agent | None = None
_spanish_agent: Agent | None = None
_english_agent: Agent | None = None
_triage_agent: Agent | None = None


def get_french_agent() -> Agent:
    global _french_agent
    if _french_agent is None:
        _french_agent = Agent(
            name="french_agent",
            instructions=french_agent_prompt.compile(),
        )
    return _french_agent


def get_spanish_agent() -> Agent:
    global _spanish_agent
    if _spanish_agent is None:
        _spanish_agent = Agent(
            name="spanish_agent",
            instructions=spanish_agent_prompt.compile(),
        )
    return _spanish_agent


def get_english_agent() -> Agent:
    global _english_agent
    if _english_agent is None:
        _english_agent = Agent(
            name="english_agent",
            instructions=english_agent_prompt.compile(),
        )
    return _english_agent


def get_triage_agent() -> Agent:
    global _triage_agent
    if _triage_agent is None:
        _triage_agent = Agent(
            name="triage_agent",
            instructions=triage_routing_agent_prompt.compile(),
            handoffs=[get_french_agent(), get_spanish_agent(), get_english_agent()],
        )
    return _triage_agent


@pixie.app
async def openai_multilingual_routing() -> pixie.PixieGenerator[str, str]:
    """
    Multi-agent language routing system with streaming responses.

    The triage agent receives messages and hands off to the appropriate language
    specialist agent (French, Spanish, or English) based on the detected language.

    Responses are streamed in real-time to provide immediate feedback.

    Yields:
        Streamed agent responses

    Receives:
        User messages in any supported language via InputRequired
    """
    agent = get_triage_agent()
    inputs: list[TResponseInputItem] = []

    yield "Hi! We speak French, Spanish and English. How can I help?"
    yield "(Type 'exit' to quit)"

    while True:
        # Get user message
        user_msg = yield pixie.InputRequired(str)

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


@pixie.app
async def openai_multilingual_routing_simple(
    initial_message: str,
) -> pixie.PixieGenerator[str, str]:
    """
    Simplified multilingual routing with single initial message.

    This version accepts an initial message and then enters an interactive loop.
    Good for testing with a specific language right away.

    Args:
        initial_message: The first message to send (in any supported language)

    Yields:
        Streamed agent responses

    Receives:
        Follow-up user messages via InputRequired
    """
    agent = get_triage_agent()
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
        user_msg = yield pixie.InputRequired(str)

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
