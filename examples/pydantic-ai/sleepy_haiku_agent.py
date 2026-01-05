"""Pause/resume demo using a pydanticAI agent with a sleep tool.

The agent flows forever:
1) take a topic from the user
2) call a tool that sleeps 5-10 seconds
3) write a haiku about the topic
4) repeat steps 2-3 indefinitely

Run the server (`pixie`) and subscribe via GraphiQL:
subscription {
  run(name: "sleep_haiku", inputData: "rain" ) {
    runId
    status
    data
    breakpoint {
      spanName
      breakpointType
      breakpointTiming
      spanAttributes
    }
  }
}

The agent can be paused/resumed at any point between tool calls.
mutation {
    pauseRun(runId: "<RUN_ID>") {
        success
    }
}

mutation {
    resumeRun(runId: "<RUN_ID>") {
        success
    }
}
"""

import asyncio
import logging
from pydantic_ai import Agent, RunContext

from pixie import pixie_app, PixieGenerator

logger = logging.getLogger(__name__)


haiku_agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt=(
        "You are a patient haiku bot. For every cycle, you must first call the "
        "`sleep_for_a_bit` tool to wait for 5 seconds, then craft a new "
        "three-line haiku about the provided topic. Keep answers concise."
    ),
)


@haiku_agent.tool
async def sleep_for_a_bit(_ctx: RunContext[None]) -> str:
    """Sleep for 5 seconds before writing a haiku."""
    await asyncio.sleep(5)
    return "Slept for 5 seconds."


@pixie_app
async def sleepy_haiku(topic: str) -> PixieGenerator[str, None]:
    """Run one sleep + haiku cycle through the agent."""
    Agent.instrument_all()

    iteration = 10

    yield f"Starting haiku cycles on topic: {topic}, repeating {iteration} times."

    for i in range(1, 10):
        result = await haiku_agent.run(topic)
        yield f"--- Haiku #{i} ---\n{result.output}\nNow let me sleep a bit..."
