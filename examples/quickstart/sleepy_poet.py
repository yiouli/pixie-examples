"""An interactive agent that writes haikus while taking naps."""

import asyncio
from dataclasses import dataclass
import logging
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModelSettings

from pixie import pixie_app, PixieGenerator, UserInputRequirement

logger = logging.getLogger(__name__)


@dataclass
class PoetDeps:
    """Dependencies for the sleepy poet agent."""

    queue: asyncio.Queue
    time_to_write: bool = False


poet = Agent(
    "openai:gpt-4o-mini",
    system_prompt=(
        "You are a sleepy poet. When given a topic, you first take a nap by calling the "
        "`sleep_for_a_bit` tool once and once only, then write a haiku about the topic."
    ),
    deps_type=PoetDeps,
    model_settings=OpenAIChatModelSettings(
        parallel_tool_calls=False  # Ensure single sleep tool call per turn
    ),
)


@poet.tool
async def sleep_for_a_bit(ctx: RunContext[PoetDeps]) -> str:
    """Sleep for 5 seconds before writing a haiku."""
    if ctx.deps.time_to_write:
        return "I've already napped, It's time to write the haiku now."

    await ctx.deps.queue.put("Let me take a nap...")
    for i in range(3):
        await asyncio.sleep(1)
        await ctx.deps.queue.put("z" * (i + 1))
    ctx.deps.time_to_write = True
    return "Poet napped for 3 seconds."


class HaikuRequest(BaseModel):
    """Input config for sleepy haiku agent."""

    topic: str = Field(..., description="The topic to write haikus about.")
    count: int = Field(3, description="Number of haikus to write.")


@pixie_app
async def example_sleepy_poet(_: None) -> PixieGenerator[HaikuRequest, str]:
    """An interactive agent that writes haikus while taking naps.

    This agent accepts a topic and number of haikus to write, then for each haiku,
    it first calls a tool to sleep for a few seconds before composing the haiku.

    Args:
        _: No input data is required to start the app.
    Yields:
        HaikuRequest: Requirement for user's input for the haiku request.
        str: The generated haikus interleaved with sleep updates.
    """

    q = asyncio.Queue[str | None]()
    deps = PoetDeps(queue=q)

    yield "I'm the best poet in town! I can write haikus on any topic."

    while True:
        yield "What's your request?"
        config = yield UserInputRequirement(HaikuRequest)

        async def write_haikus():
            for i in range(0, config.count):
                result = await poet.run(config.topic, deps=deps)
                await q.put(f"### Haiku #{i+1}\n{result.output}\n")
                deps.time_to_write = False  # Reset for next haiku

            await q.put(None)  # Signal completion

        task = asyncio.create_task(write_haikus())
        try:
            while True:
                update = await q.get()
                if update is None:
                    break
                yield update

        except asyncio.CancelledError:
            task.cancel()
            return
