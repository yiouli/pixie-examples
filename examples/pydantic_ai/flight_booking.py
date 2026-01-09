"""Example of a multi-agent flow where one agent delegates work to another.

This demonstrates agent delegation and programmatic agent hand-off patterns.
"""

import datetime
from dataclasses import dataclass
from typing import Literal
from types import NoneType

from pydantic import BaseModel, Field
from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRetry,
    RunContext,
    RunUsage,
    UsageLimits,
)
from pixie import pixie_app, PixieGenerator, UserInputRequirement


class FlightDetails(BaseModel):
    """Details of the most suitable flight."""

    flight_number: str
    price: int
    origin: str = Field(description="Three-letter airport code")
    destination: str = Field(description="Three-letter airport code")
    date: datetime.date


class NoFlightFound(BaseModel):
    """When no valid flight is found."""


@dataclass
class Deps:
    web_page_text: str
    req_origin: str
    req_destination: str
    req_date: datetime.date


# This agent is responsible for controlling the flow of the conversation.
search_agent = Agent[Deps, FlightDetails | NoFlightFound](
    "openai:gpt-4o-mini",
    output_type=FlightDetails | NoFlightFound,  # type: ignore
    retries=4,
    system_prompt=(
        "Your job is to find the cheapest flight for the user on the given date. "
    ),
)


# This agent is responsible for extracting flight details from web page text.
extraction_agent = Agent(
    "openai:gpt-4o-mini",
    output_type=list[FlightDetails],
    system_prompt="Extract all the flight details from the given text.",
)


@search_agent.tool
async def extract_flights(ctx: RunContext[Deps]) -> list[FlightDetails]:
    """Get details of all flights."""
    # we pass the usage to the search agent so requests within this agent are counted
    agent_result = await extraction_agent.run(ctx.deps.web_page_text, usage=ctx.usage)
    return agent_result.output


@search_agent.output_validator
async def validate_output(
    ctx: RunContext[Deps], output: FlightDetails | NoFlightFound
) -> FlightDetails | NoFlightFound:
    """Procedural validation that the flight meets the constraints."""
    if isinstance(output, NoFlightFound):
        return output

    errors: list[str] = []
    if output.origin != ctx.deps.req_origin:
        errors.append(
            f"Flight should have origin {ctx.deps.req_origin}, not {output.origin}"
        )
    if output.destination != ctx.deps.req_destination:
        errors.append(
            f"Flight should have destination {ctx.deps.req_destination}, not {output.destination}"
        )
    if output.date != ctx.deps.req_date:
        errors.append(f"Flight should be on {ctx.deps.req_date}, not {output.date}")

    if errors:
        raise ModelRetry("\n".join(errors))
    else:
        return output


class SeatPreference(BaseModel):
    row: int = Field(ge=1, le=30)
    seat: Literal["A", "B", "C", "D", "E", "F"]


class Failed(BaseModel):
    """Unable to extract a seat selection."""


# This agent is responsible for extracting the user's seat selection
seat_preference_agent = Agent[None, SeatPreference | Failed](
    "openai:gpt-4o-mini",
    output_type=SeatPreference | Failed,
    system_prompt=(
        "Extract the user's seat preference. "
        "Seats A and F are window seats. "
        "Row 1 is the front row and has extra leg room. "
        "Rows 14, and 20 also have extra leg room. "
    ),
)


# in reality this would be downloaded from a booking site,
# potentially using another agent to navigate the site
flights_web_page = """
1. Flight SFO-AK123
- Price: $350
- Origin: San Francisco International Airport (SFO)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 10, 2025

2. Flight SFO-AK456
- Price: $370
- Origin: San Francisco International Airport (SFO)
- Destination: Fairbanks International Airport (FAI)
- Date: January 10, 2025

3. Flight SFO-AK789
- Price: $400
- Origin: San Francisco International Airport (SFO)
- Destination: Juneau International Airport (JNU)
- Date: January 20, 2025

4. Flight NYC-LA101
- Price: $250
- Origin: San Francisco International Airport (SFO)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 10, 2025

5. Flight CHI-MIA202
- Price: $200
- Origin: Chicago O'Hare International Airport (ORD)
- Destination: Miami International Airport (MIA)
- Date: January 12, 2025

6. Flight BOS-SEA303
- Price: $120
- Origin: Boston Logan International Airport (BOS)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 12, 2025

7. Flight DFW-DEN404
- Price: $150
- Origin: Dallas/Fort Worth International Airport (DFW)
- Destination: Denver International Airport (DEN)
- Date: January 10, 2025

8. Flight ATL-HOU505
- Price: $180
- Origin: Hartsfield-Jackson Atlanta International Airport (ATL)
- Destination: George Bush Intercontinental Airport (IAH)
- Date: January 10, 2025
"""

# restrict how many requests this app can make to the LLM
usage_limits = UsageLimits(request_limit=15)


@pixie_app
async def flight_booking(_: NoneType) -> PixieGenerator[str, str]:
    """Multi-agent flight booking with search, extraction, and seat selection.

    This example demonstrates:
    - Agent delegation (search_agent -> extraction_agent)
    - Programmatic agent hand-off (search -> seat selection)
    - Interactive user input for flight confirmation and seat preference
    """
    Agent.instrument_all()

    deps = Deps(
        web_page_text=flights_web_page,
        req_origin="SFO",
        req_destination="ANC",
        req_date=datetime.date(2025, 1, 10),
    )
    message_history: list[ModelMessage] | None = None
    usage: RunUsage = RunUsage()

    yield "🛫 Welcome to Flight Booking! Searching for flights..."

    # Flight search loop
    while True:
        agent_result = await search_agent.run(
            f"Find me a flight from {deps.req_origin} to {deps.req_destination} on {deps.req_date}",
            deps=deps,
            usage=usage,
            message_history=message_history,
            usage_limits=usage_limits,
        )

        if isinstance(agent_result.output, NoFlightFound):
            yield "❌ No flight found. Ending search."
            break
        else:
            flight = agent_result.output
            yield f"✈️ Flight found: {flight.flight_number}"
            yield f"   Price: ${flight.price}"
            yield f"   Route: {flight.origin} → {flight.destination}"
            yield f"   Date: {flight.date}"
            yield "\nDo you want to buy this flight, or keep searching? (buy/search)"

            answer = yield UserInputRequirement(str)
            answer = answer.lower().strip()

            if answer == "buy":
                # Move to seat selection
                yield "\n🪑 Great! Now let's select your seat..."
                seat_message_history: list[ModelMessage] | None = None

                while True:
                    yield 'What seat would you like? (e.g., "row 1 seat A" or "window seat with leg room")'
                    seat_input = yield UserInputRequirement(str)

                    seat_result = await seat_preference_agent.run(
                        seat_input,
                        message_history=seat_message_history,
                        usage=usage,
                        usage_limits=usage_limits,
                    )

                    if isinstance(seat_result.output, SeatPreference):
                        seat = seat_result.output
                        yield "\n✅ Booking confirmed!"
                        yield f"   Flight: {flight.flight_number}"
                        yield f"   Seat: Row {seat.row}, Seat {seat.seat}"
                        yield f"   Total Cost: ${flight.price}"
                        return
                    else:
                        yield "❌ Could not understand seat preference. Please try again."
                        seat_message_history = seat_result.all_messages()
            else:
                yield "\n🔄 Searching for another flight..."
                message_history = agent_result.all_messages(
                    output_tool_return_content="Please suggest another flight"
                )


# For local testing
async def test():
    """Test function for local development."""
    async for message in flight_booking(None):
        if isinstance(message, UserInputRequirement):
            # Simulate user input for testing
            if "buy" in str(message):
                message = "buy"
            elif "seat" in str(message):
                message = "row 1 seat A"
            else:
                message = "search"
        print(message)


if __name__ == "__main__":
    import asyncio

    asyncio.run(test())
