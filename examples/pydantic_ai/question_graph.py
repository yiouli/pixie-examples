"""Example of a graph-based state machine for asking and evaluating questions.

This demonstrates using pydantic_graph for complex control flow.
"""

from dataclasses import dataclass, field

from pydantic import BaseModel
from pydantic_ai import Agent, ModelMessage, format_as_xml
from pydantic_graph import (
    BaseNode,
    End,
    Graph,
    GraphRunContext,
)
import pixie

# Agent for asking questions
ask_agent = Agent("openai:gpt-4o-mini", output_type=str)


@dataclass
class QuestionState:
    question: str | None = None
    answer: str | None = None
    ask_agent_messages: list[ModelMessage] = field(default_factory=list)
    evaluate_agent_messages: list[ModelMessage] = field(default_factory=list)


@dataclass
class Ask(BaseNode[QuestionState]):
    """Generate a question using the AI."""

    async def run(self, ctx: GraphRunContext[QuestionState]) -> "Answer":
        agent_result = await ask_agent.run(
            "Ask a simple question with a single correct answer.",
            message_history=ctx.state.ask_agent_messages,
        )
        ctx.state.ask_agent_messages += agent_result.all_messages()
        ctx.state.question = agent_result.output
        ctx.state.answer = None
        return Answer(agent_result.output)


@dataclass
class Answer(BaseNode[QuestionState]):
    """Wait for user to provide an answer."""

    question: str

    async def run(self, ctx: GraphRunContext[QuestionState]) -> "Evaluate":
        if ctx.state.answer is None:
            raise ValueError("Answer need to be set after Ask() node in the main loop.")
        return Evaluate(ctx.state.answer)


class EvaluationOutput(BaseModel, use_attribute_docstrings=True):
    correct: bool
    """Whether the answer is correct."""
    comment: str
    """Comment on the answer, reprimand the user if the answer is wrong."""


# Agent for evaluating answers
evaluate_agent = Agent(
    "openai:gpt-4o-mini",
    output_type=EvaluationOutput,
    system_prompt="Given a question and answer, evaluate if the answer is correct.",
)


@dataclass
class Evaluate(BaseNode[QuestionState, None, str]):
    """Evaluate the user's answer."""

    answer: str

    async def run(
        self,
        ctx: GraphRunContext[QuestionState],
    ) -> End[str] | "Reprimand":
        assert ctx.state.question is not None
        agent_result = await evaluate_agent.run(
            format_as_xml({"question": ctx.state.question, "answer": self.answer}),
            message_history=ctx.state.evaluate_agent_messages,
        )
        ctx.state.evaluate_agent_messages += agent_result.all_messages()
        if agent_result.output.correct:
            return End(agent_result.output.comment)
        else:
            return Reprimand(agent_result.output.comment)


@dataclass
class Reprimand(BaseNode[QuestionState]):
    """Tell the user they got it wrong and ask another question."""

    comment: str

    async def run(self, ctx: GraphRunContext[QuestionState]) -> Ask:
        ctx.state.question = None
        return Ask()


# Create the question graph
_question_graph = Graph(
    nodes=(Ask, Answer, Evaluate, Reprimand), state_type=QuestionState
)


@pixie.pixie_app
async def question_graph() -> pixie.PixieGenerator[str, str]:
    """Interactive Q&A game using graph-based state machine.

    The AI asks questions, the user answers, and the AI evaluates.
    If wrong, user gets reprimanded and a new question is asked.
    """

    yield "🧠 Welcome to the Q&A Challenge!"
    yield "I'll ask you questions and evaluate your answers.\n"

    state = QuestionState()

    # Use the graph's run_sync method which handles node execution
    async with _question_graph.iter(Ask(), state=state) as graph_ctx:
        while True:
            node = await graph_ctx.next()

            # Handle different node types
            if isinstance(node, Ask):
                yield "🤔 Generating a question..."

            elif isinstance(node, Answer):
                # Ask user for their answer
                yield f"\n❓ Question: {node.question}"
                yield "What is your answer?"
                state.answer = yield pixie.UserInputRequirement(str)
                # Continue with the user's answer by passing it to the next iteration
                # The graph will automatically move to Evaluate with this answer
                continue

            elif isinstance(node, Evaluate):
                yield "⏳ Evaluating your answer..."

            elif isinstance(node, Reprimand):
                yield f"\n❌ {node.comment}"
                yield "\n🔄 Let's try another question...\n"

            elif isinstance(node, End):
                yield f"\n✅ {node.data}"
                yield "\n🎉 Congratulations! You got it right!"
                return


# For local testing
async def test():
    """Test function for local development."""
    async for message in question_graph(None):
        if isinstance(message, pixie.UserInputRequirement):
            # Simulate user input for testing
            message = "Paris"  # Assuming a capital question
        print(message)


if __name__ == "__main__":
    import asyncio

    asyncio.run(test())
