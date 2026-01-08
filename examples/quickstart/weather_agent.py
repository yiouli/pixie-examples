"""Weather agent example with multiple tools.

This example demonstrates:
- Multiple tools that need to be called in sequence
- Agent dependencies for HTTP client
- Streaming text responses
- Integration with Pixie SDK

The agent uses two tools:
1. get_lat_lng: Get latitude/longitude from location description
2. get_weather: Get weather data for given coordinates

Run with:
    poetry run pixie

Then query via GraphQL:
    subscription {
      run(name: "weather_agent", inputData: "What is the weather in London?") {
        runId
        status
        data
      }
    }
"""

import asyncio
from dataclasses import dataclass
from typing import Any
import logfire
from httpx import AsyncClient
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pixie import pixie_app

# Configure logfire
logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()


@dataclass
class Deps:
    """Dependencies for the weather agent."""

    client: AsyncClient


class LatLng(BaseModel):
    """Latitude and longitude coordinates."""

    lat: float
    lng: float


# Create the weather agent
_weather_agent = Agent(
    "openai:gpt-4o-mini",
    instructions="Be concise, reply with one sentence.",
    deps_type=Deps,
    retries=2,
)


@_weather_agent.tool
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


@_weather_agent.tool
async def get_weather(ctx: RunContext[Deps], lat: float, lng: float) -> dict[str, Any]:
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


@pixie_app
async def weather_agent(query: str) -> str:
    """Get weather information for a location.

    Args:
        query: Natural language query about weather (e.g., "What is the weather in London?")

    Returns:
        str: Weather information
    """
    # Enable instrumentation
    Agent.instrument_all()

    # Create HTTP client and dependencies
    async with AsyncClient() as client:
        logfire.instrument_httpx(client, capture_all=True)
        deps = Deps(client=client)

        # Run the agent
        agent_result = await _weather_agent.run(query, deps=deps)

        return agent_result.output


if __name__ == "__main__":
    # For testing locally
    async def test():
        output = await weather_agent(
            "What is the weather like in London and in Wiltshire?"
        )
        print("Response:", output)

    asyncio.run(test())
