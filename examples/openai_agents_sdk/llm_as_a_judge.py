"""
LLM as a Judge Pattern - Pixie Integration

This example demonstrates the LLM as a judge pattern where one agent generates content
and another agent evaluates it iteratively until quality standards are met.

Pattern: Multi-Step Workflow
Original: https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/llm_as_a_judge.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agents import Agent, ItemHelpers, Runner, TResponseInputItem
import pixie.sdk as pixie


@dataclass
class EvaluationFeedback:
    """Feedback from the evaluator agent"""

    feedback: str
    score: Literal["pass", "needs_improvement", "fail"]


generator_prompt = pixie.create_prompt(
    "story_outline_generator",
    description="Generates short story outlines based on user input",
)
evaluator_prompt = pixie.create_prompt(
    "evaluator",
    description="Evaluates story outlines and provides improvement feedback",
)


@pixie.app
async def openai_agents_llm_as_a_judge(topic: str) -> pixie.PixieGenerator[str, None]:
    """
    Generate a story outline using LLM-as-a-judge pattern.

    The story generator creates an outline, which is then evaluated by a judge agent.
    The process repeats until the judge is satisfied with the quality.

    Args:
        topic: The story topic to generate an outline for

    Yields:
        Status updates and the final story outline
    """
    yield f"Starting story generation for: {topic}"

    # Story outline generator agent
    story_outline_generator = Agent(
        name=generator_prompt.id,
        instructions=generator_prompt.compile(),
    )

    # Evaluator agent that judges the story outline
    evaluator = Agent[None](
        name=evaluator_prompt.id,
        instructions=evaluator_prompt.compile(),
        output_type=EvaluationFeedback,
    )

    input_items: list[TResponseInputItem] = [{"content": topic, "role": "user"}]
    latest_outline: str | None = None

    iteration = 0
    max_iterations = 10

    while iteration < max_iterations:
        iteration += 1
        yield f"\n--- Iteration {iteration} ---"

        # Generate story outline
        yield "Generating story outline..."
        story_outline_result = await Runner.run(
            story_outline_generator,
            input_items,
        )
        input_items = story_outline_result.to_input_list()
        latest_outline = ItemHelpers.text_message_outputs(
            story_outline_result.new_items
        )
        yield f"Story outline generated:\n{latest_outline}"

        # Evaluate the outline
        yield "\nEvaluating outline..."
        evaluator_result = await Runner.run(evaluator, input_items)
        result: EvaluationFeedback = evaluator_result.final_output

        yield f"Evaluator score: {result.score}"

        if result.score == "pass":
            yield "\n✓ Story outline is good enough!"
            break

        # Provide feedback for next iteration
        yield f"Feedback: {result.feedback}"
        yield "Re-running with feedback..."
        input_items.append({"content": f"Feedback: {result.feedback}", "role": "user"})

    if iteration >= max_iterations:
        yield f"\n⚠ Reached maximum iterations ({max_iterations})"

    yield f"\n=== Final Story Outline ===\n{latest_outline}"
