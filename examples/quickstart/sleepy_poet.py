"""An interactive agent that writes haikus while taking naps."""

import asyncio
from dataclasses import dataclass
import logging
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModelSettings

import pixie.sdk as pixie

sleepy_poet_prompt = pixie.create_prompt(
    "sleepy_poet",
    description="A sleepy poet that naps before writing haikus",
)

logger = logging.getLogger(__name__)


@dataclass
class PoetDeps:
    """Dependencies for the sleepy poet agent."""

    queue: asyncio.Queue
    time_to_write: bool = False


class HaikuRequest(BaseModel):
    """Input config for sleepy haiku agent."""

    topic: str = Field(..., description="The topic to write haikus about.")
    count: int = Field(3, description="Number of haikus to write.")


@pixie.app
async def example_sleepy_poet() -> pixie.PixieGenerator[str, HaikuRequest]:
    """An interactive agent that writes haikus while taking naps.

    This agent accepts a topic and number of haikus to write, then for each haiku,
    it first calls a tool to sleep for a few seconds before composing the haiku.

    Yields:
        HaikuRequest: Requirement for user's input for the haiku request.
        str: The generated haikus interleaved with sleep updates.
    """

    poet = Agent(
        "openai:gpt-4o-mini",
        system_prompt=sleepy_poet_prompt.compile(),
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

    q = asyncio.Queue[str | None]()
    deps = PoetDeps(queue=q)

    yield "I'm the best poet in town! I can write haikus on any topic."

    while True:
        yield "What's your request?"
        config = yield pixie.InputRequired(HaikuRequest)

        ack = asyncio.Event()
        haiku_writen = False

        async def write_haikus():
            for i in range(0, config.count):
                print("writing haiku", i + 1)
                result = await poet.run(config.topic, deps=deps)
                await q.put(f"### Haiku #{i+1}\n{result.output}\n")
                deps.time_to_write = False  # Reset for next haiku

                nonlocal haiku_writen
                haiku_writen = True  # let consumer know a kaiku was written

                await ack.wait()  # Wait for acknowledgment before next haiku
                ack.clear()

            await q.put(None)  # Signal completion

        task = asyncio.create_task(write_haikus())
        try:
            while True:
                update = await q.get()
                if update is None:
                    break
                print(update)
                yield update
                if haiku_writen:  # acknowledge after receiving a writen haiku
                    haiku_writen = False
                    ack.set()  # send acknowledgment to writer so it can proceed

        except asyncio.CancelledError:
            task.cancel()
            return
