"""Interactive weather agent."""

import asyncio
from dataclasses import dataclass
from typing import Any
from httpx import AsyncClient
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
import pixie.sdk as pixie


@dataclass
class Deps:
    """Dependencies for the weather agent."""

    client: AsyncClient


class LatLng(BaseModel):
    """Latitude and longitude coordinates."""

    lat: float
    lng: float


pydantic_weather_agent_prompt = pixie.create_prompt(
    "pydantic_weather_agent",
    description="Prompt for an interactive agent built with Pydantic-ai.",
)


@pixie.app
async def example_weather_agent() -> pixie.PixieGenerator[str, str]:
    """Interactive weather agent.

    This agent interacts with the user to get weather information for a specified location.
    It uses tools to fetch latitude/longitude and weather data, and supports a multi-turn
    conversation with the user.

    The agent will:
    - Prompt the user for a location.
    - Fetch latitude and longitude for the location.
    - Retrieve weather data for the coordinates.
    - Display the weather information to the user.

    The conversation continues until the user enters a stop word ('exit', 'quit', or 'stop').
    """

    # Create the weather agent
    agent = Agent(
        "openai:gpt-4o-mini",
        instructions=pydantic_weather_agent_prompt.compile(),
        deps_type=Deps,
        retries=2,
    )

    @agent.tool
    async def get_lat_lng(ctx: RunContext[Deps], location_description: str) -> LatLng:
        """Get the latitude and longitude of a location.

        Args:
            ctx: The context.
            location_description: A description of a location.
        """
        # NOTE: Uses demo endpoints that return random data
        r = await ctx.deps.client.get(
            "https://demo-endpoints.pydantic.workers.dev/latlng",
            params={"location": location_description},
        )
        r.raise_for_status()
        return LatLng.model_validate_json(r.content)

    @agent.tool
    async def get_weather(
        ctx: RunContext[Deps], lat: float, lng: float
    ) -> dict[str, Any]:
        """Get the weather at a location.

        Args:
            ctx: The context.
            lat: Latitude of the location.
            lng: Longitude of the location.
        """
        # NOTE: Uses demo endpoints that return random data
        temp_response, descr_response = await asyncio.gather(
            ctx.deps.client.get(
                "https://demo-endpoints.pydantic.workers.dev/number",
                params={"min": 10, "max": 30},
            ),
            ctx.deps.client.get(
                "https://demo-endpoints.pydantic.workers.dev/weather",
                params={"lat": lat, "lng": lng},
            ),
        )
        temp_response.raise_for_status()
        descr_response.raise_for_status()
        return {
            "temperature": f"{temp_response.text} °C",
            "description": descr_response.text,
        }

    yield "Hi! I can help you find the weather for any location."
    yield "Enter 'exit', 'quit', or 'stop' to end the conversation."

    # Create HTTP client and dependencies
    async with AsyncClient() as client:
        deps = Deps(client=client)

        user_message = None
        stop_words = ["exit", "quit", "stop"]
        while user_message not in stop_words:
            yield "What location would you like the weather for?"
            user_message = yield pixie.InputRequired(str)
            result = await agent.run(user_message, deps=deps)
            yield result.output
