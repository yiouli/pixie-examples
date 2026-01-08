from typing import Literal
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

# from pixie import pixie_app  # type: ignore
from pixie import pixie_app


# Define the weather tool result model
class WeatherResult(BaseModel):
    location: str
    temperature: float
    conditions: Literal["sunny", "cloudy", "rainy", "snowy"]
    humidity: int


# Create a simple weather agent
weather_agent = Agent(
    "openai:gpt-4o",
    system_prompt="You are a helpful weather assistant. Provide weather information based on the location requested.",
)


@weather_agent.tool
async def get_weather(_ctx: RunContext[None], location: str) -> WeatherResult:
    """Get the current weather for a location (hardcoded for demo)."""
    # Hardcoded weather data
    weather_data = {
        "san francisco": WeatherResult(
            location="San Francisco",
            temperature=18.5,
            conditions="cloudy",
            humidity=75,
        ),
        "new york": WeatherResult(
            location="New York",
            temperature=22.0,
            conditions="sunny",
            humidity=60,
        ),
        "london": WeatherResult(
            location="London",
            temperature=12.0,
            conditions="rainy",
            humidity=85,
        ),
    }

    # Return hardcoded data or default
    return weather_data.get(
        location.lower(),
        WeatherResult(
            location=location,
            temperature=20.0,
            conditions="sunny",
            humidity=50,
        ),
    )


@pixie_app
async def weather(query: str) -> str:
    """Run the weather agent with the given query and return a string summary."""
    Agent.instrument_all()
    result = await weather_agent.run(query)
    return result.output
