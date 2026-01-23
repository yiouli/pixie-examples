from pydantic import Field
from pydantic_ai import Agent

import pixie.sdk as pixie


class Variables(pixie.Variables):
    problem_description: str = Field(
        ..., description="A description of the problem to be solved."
    )


agent_prompt = pixie.create_prompt(
    "problem_solver",
    Variables,
    description="Problem solving agent that thinks step by step",
)


@pixie.app
async def example_problem_solver(problem_description: str):
    """A problem solving agent that thinks and solves complex problems.

    This agent utilizes pixie's prompt libraries so you can do prompt optimization within the UI.

    Args:
        problem_description (str): A description of the problem to be solved.
    """

    agent = Agent(
        name=agent_prompt.id,
        instructions=agent_prompt.compile(
            Variables(problem_description=problem_description)
        ),
        model="gpt-4o-mini",
    )
    response = await agent.run()
    yield response.output
