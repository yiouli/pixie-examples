"""
Customer Service - Multi-Agent & Multi-Turn Chatbot - Pixie Integration

This example demonstrates a customer service system with multiple specialized agents
that can hand off to each other based on customer needs.

Pattern: Multi-Agent & Multi-Turn
Original: https://github.com/openai/openai-agents-python/blob/main/examples/customer_service/main.py
"""

from __future__ import annotations

import random
from types import NoneType

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
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from pixie import pixie_app, PixieGenerator, UserInputRequirement


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
# AGENTS
# ============================================================================

faq_agent = Agent[AirlineAgentContext](
    name="FAQ Agent",
    handoff_description="A helpful agent that can answer questions about the airline.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

    You are an FAQ agent. If you are speaking to a customer, you probably were
    transferred to from the triage agent.

    Use the following routine to support the customer.

    # Routine
    1. Identify the last question asked by the customer.
    2. Use the faq lookup tool to answer the question. Do not rely on your own knowledge.
    3. If you cannot answer the question, transfer back to the triage agent.""",
    tools=[faq_lookup_tool],
)

seat_booking_agent = Agent[AirlineAgentContext](
    name="Seat Booking Agent",
    handoff_description="A helpful agent that can update a seat on a flight.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

    You are a seat booking agent. If you are speaking to a customer, you probably were
    transferred to from the triage agent.

    Use the following routine to support the customer.

    # Routine
    1. Ask for their confirmation number.
    2. Ask the customer what their desired seat number is.
    3. Use the update seat tool to update the seat on the flight.

    If the customer asks a question that is not related to the routine, transfer back to the
    triage agent.""",
    tools=[update_seat],
)

triage_agent = Agent[AirlineAgentContext](
    name="Triage Agent",
    handoff_description="A triage agent that can delegate a customer's request to the appropriate agent.",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} "
        "You are a helpful triaging agent. You can use your tools to delegate questions to other appropriate agents."
    ),
    handoffs=[
        faq_agent,
        handoff(agent=seat_booking_agent, on_handoff=on_seat_booking_handoff),
    ],
)

# Set up bidirectional handoffs
faq_agent.handoffs.append(triage_agent)
seat_booking_agent.handoffs.append(triage_agent)


@pixie_app
async def airline_customer_service(_: NoneType) -> PixieGenerator[str, str]:
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
        User messages via UserInputRequirement
    """
    current_agent: Agent[AirlineAgentContext] = triage_agent
    input_items: list[TResponseInputItem] = []
    context = AirlineAgentContext()

    yield "Welcome to Airline Customer Service! How can I help you today?"
    yield "(Type 'exit', 'quit', or 'bye' to end the conversation)"

    while True:
        # Get user input
        user_input = yield UserInputRequirement(str)

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
