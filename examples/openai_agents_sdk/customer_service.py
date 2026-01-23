"""
Customer Service - Multi-Agent & Multi-Turn Chatbot - Pixie Integration

This example demonstrates a customer service system with multiple specialized agents
that can hand off to each other based on customer needs.

Pattern: Multi-Agent & Multi-Turn
Original: https://github.com/openai/openai-agents-python/blob/main/examples/customer_service/main.py
"""

from __future__ import annotations

import random

from pydantic import BaseModel
from agents import (
    Agent,
    HandoffOutputItem,
    ItemHelpers,
    MessageOutputItem,
    RunContextWrapper,
    Runner,
    ToolCallItem,
    ToolCallOutputItem,
    TResponseInputItem,
    function_tool,
    handoff,
)
import pixie.sdk as pixie


faq_agent_prompt = pixie.create_prompt(
    "airline_faq_agent",
    description="FAQ agent that answers customer questions using knowledge base lookup",
)
seat_booking_agent_prompt = pixie.create_prompt(
    "airline_seat_booking_agent",
    description="Seat booking agent that updates flight seat assignments",
)
triage_agent_prompt = pixie.create_prompt(
    "airline_triage_agent",
    description="Triage agent that delegates customer inquiries to appropriate specialists",
)


# ============================================================================
# CONTEXT
# ============================================================================


class AirlineAgentContext(BaseModel):
    """Shared context across all airline customer service agents"""

    passenger_name: str | None = None
    confirmation_number: str | None = None
    seat_number: str | None = None
    flight_number: str | None = None


# ============================================================================
# TOOLS
# ============================================================================


@function_tool(
    name_override="faq_lookup_tool",
    description_override="Lookup frequently asked questions.",
)
async def faq_lookup_tool(question: str) -> str:
    """Simulates FAQ lookup for common airline questions"""
    question_lower = question.lower()

    if any(
        keyword in question_lower
        for keyword in [
            "bag",
            "baggage",
            "luggage",
            "carry-on",
            "hand luggage",
            "hand carry",
        ]
    ):
        return (
            "You are allowed to bring one bag on the plane. "
            "It must be under 50 pounds and 22 inches x 14 inches x 9 inches."
        )
    elif any(
        keyword in question_lower for keyword in ["seat", "seats", "seating", "plane"]
    ):
        return (
            "There are 120 seats on the plane. "
            "There are 22 business class seats and 98 economy seats. "
            "Exit rows are rows 4 and 16. "
            "Rows 5-8 are Economy Plus, with extra legroom."
        )
    elif any(
        keyword in question_lower
        for keyword in [
            "wifi",
            "internet",
            "wireless",
            "connectivity",
            "network",
            "online",
        ]
    ):
        return "We have free wifi on the plane, join Airline-Wifi"

    return "I'm sorry, I don't know the answer to that question."


@function_tool
async def update_seat(
    context: RunContextWrapper[AirlineAgentContext],
    confirmation_number: str,
    new_seat: str,
) -> str:
    """
    Update the seat for a given confirmation number.

    Args:
        confirmation_number: The confirmation number for the flight.
        new_seat: The new seat to update to.
    """
    # Update the context based on the customer's input
    context.context.confirmation_number = confirmation_number
    context.context.seat_number = new_seat

    # Ensure that the flight number has been set by the incoming handoff
    assert context.context.flight_number is not None, "Flight number is required"

    return f"Updated seat to {new_seat} for confirmation number {confirmation_number}"


# ============================================================================
# HOOKS
# ============================================================================


async def on_seat_booking_handoff(
    context: RunContextWrapper[AirlineAgentContext],
) -> None:
    """Generate a flight number when handing off to seat booking agent"""
    flight_number = f"FLT-{random.randint(100, 999)}"
    context.context.flight_number = flight_number


# ============================================================================
# AGENTS (Lazy initialization to avoid compile() at module import time)
# ============================================================================

_faq_agent: Agent[AirlineAgentContext] | None = None
_seat_booking_agent: Agent[AirlineAgentContext] | None = None
_triage_agent: Agent[AirlineAgentContext] | None = None
_agents_initialized: bool = False


def _initialize_agents() -> None:
    """Initialize all agents with proper handoffs."""
    global _faq_agent, _seat_booking_agent, _triage_agent, _agents_initialized

    if _agents_initialized:
        return

    _faq_agent = Agent[AirlineAgentContext](
        name="FAQ Agent",
        handoff_description="A helpful agent that can answer questions about the airline.",
        instructions=faq_agent_prompt.compile(),
        tools=[faq_lookup_tool],
    )

    _seat_booking_agent = Agent[AirlineAgentContext](
        name="Seat Booking Agent",
        handoff_description="A helpful agent that can update a seat on a flight.",
        instructions=seat_booking_agent_prompt.compile(),
        tools=[update_seat],
    )

    _triage_agent = Agent[AirlineAgentContext](
        name="Triage Agent",
        handoff_description="A triage agent that can delegate a customer's request to the appropriate agent.",
        instructions=triage_agent_prompt.compile(),
        handoffs=[
            _faq_agent,
            handoff(agent=_seat_booking_agent, on_handoff=on_seat_booking_handoff),
        ],
    )

    # Set up bidirectional handoffs
    _faq_agent.handoffs.append(_triage_agent)
    _seat_booking_agent.handoffs.append(_triage_agent)

    _agents_initialized = True


def get_triage_agent() -> Agent[AirlineAgentContext]:
    """Get the triage agent (entry point)."""
    _initialize_agents()
    assert _triage_agent is not None
    return _triage_agent


@pixie.app
async def openai_agents_airline_customer_service() -> pixie.PixieGenerator[str, str]:
    """
    Multi-agent customer service chatbot for an airline.

    This system uses three specialized agents:
    - Triage Agent: Routes customer requests to appropriate agents
    - FAQ Agent: Answers common questions using FAQ lookup
    - Seat Booking Agent: Handles seat change requests

    Agents can hand off to each other based on customer needs.

    Yields:
        Agent responses and status updates

    Receives:
        User messages via InputRequired
    """
    current_agent: Agent[AirlineAgentContext] = get_triage_agent()
    input_items: list[TResponseInputItem] = []
    context = AirlineAgentContext()

    yield "Welcome to Airline Customer Service! How can I help you today?"
    yield "(Type 'exit', 'quit', or 'bye' to end the conversation)"

    while True:
        # Get user input
        user_input = yield pixie.InputRequired(str)

        # Check for exit commands
        if user_input.lower() in {"exit", "quit", "bye"}:
            yield "Thank you for contacting us. Have a great flight!"
            break

        # Add user message to input
        input_items.append({"content": user_input, "role": "user"})

        # Run the current agent
        result = await Runner.run(current_agent, input_items, context=context)

        # Process and yield all new items from the result
        for new_item in result.new_items:
            agent_name = new_item.agent.name

            if isinstance(new_item, MessageOutputItem):
                message_text = ItemHelpers.text_message_output(new_item)
                yield f"{agent_name}: {message_text}"

            elif isinstance(new_item, HandoffOutputItem):
                handoff_msg = (
                    f"[Handed off from {new_item.source_agent.name} "
                    f"to {new_item.target_agent.name}]"
                )
                yield handoff_msg

            elif isinstance(new_item, ToolCallItem):
                yield f"{agent_name}: [Calling a tool...]"

            elif isinstance(new_item, ToolCallOutputItem):
                yield f"{agent_name}: [Tool result: {new_item.output}]"

        # Update state for next iteration
        input_items = result.to_input_list()
        current_agent = result.last_agent
