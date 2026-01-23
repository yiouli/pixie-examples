"""Simple example of using Pydantic AI to construct a Pydantic model from a text input.

This example demonstrates:
- Using Pydantic AI with structured output
- Integration with Pixie SDK for observability

Run with:
    poetry run pixie

Then query via GraphQL:
    subscription {
      run(name: "pydantic_model_example", inputData: "The windy city in the US of A.") {
        runId
        status
        data
      }
    }
"""

import os
from pydantic import BaseModel
from pydantic_ai import Agent
import pixie.sdk as pixie


class MyModel(BaseModel):
    """Model representing a city and country."""

    city: str
    country: str


# Create the agent
model = os.getenv("PYDANTIC_AI_MODEL", "openai:gpt-4o-mini")
agent = Agent(model, output_type=MyModel)


@pixie.app
async def pydantic_ai_structured_output(query: str) -> MyModel:
    """Extract city and country information from a text query.

    Args:
        query: Natural language text describing a city

    Returns:
        MyModel with extracted city and country
    """

    # Run the agent
    agent_result = await agent.run(query)

    # Return the structured output
    return agent_result.output


if __name__ == "__main__":
    # For testing locally
    import asyncio

    async def test():
        output = await pydantic_ai_structured_output("The windy city in the US of A.")
        print(output)

    asyncio.run(test())
