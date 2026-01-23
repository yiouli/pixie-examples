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
import pixie.sdk as pixie


class EvaluationOutput(BaseModel, use_attribute_docstrings=True):
    correct: bool
    """Whether the answer is correct."""
    comment: str
    """Comment on the answer, reprimand the user if the answer is wrong."""


ask_agent_prompt = pixie.create_prompt(
    "question_ask_agent",
    description="Asks simple questions with a single correct answer",
)
evaluate_agent_prompt = pixie.create_prompt(
    "question_evaluate_agent",
    description="Evaluates if an answer to a question is correct",
)

# Agents will be initialized lazily inside the app handler
ask_agent: Agent
evaluate_agent: Agent[None, EvaluationOutput]


def get_ask_agent() -> Agent:
    global ask_agent
    if ask_agent is None:
        ask_agent = Agent(
            "openai:gpt-4o-mini", system_prompt=ask_agent_prompt.compile()
        )
    return ask_agent


def get_evaluate_agent() -> Agent[None, EvaluationOutput]:
    global evaluate_agent
    if evaluate_agent is None:
        evaluate_agent = Agent(
            "openai:gpt-4o-mini",
            output_type=EvaluationOutput,
            system_prompt=evaluate_agent_prompt.compile(),
        )
    return evaluate_agent


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
        agent_result = await get_ask_agent().run(
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


@dataclass
class Evaluate(BaseNode[QuestionState, None, str]):
    """Evaluate the user's answer."""

    answer: str

    async def run(
        self,
        ctx: GraphRunContext[QuestionState],
    ) -> End[str] | "Reprimand":
        assert ctx.state.question is not None
        agent_result = await get_evaluate_agent().run(
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


@pixie.app
async def pydantic_ai_question_graph() -> pixie.PixieGenerator[str, str]:
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
                state.answer = yield pixie.InputRequired(str)
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
