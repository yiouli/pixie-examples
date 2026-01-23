from pydantic_ai import Agent

import pixie.sdk as pixie


prompt = pixie.create_prompt(
    "example_chatbot",
    description="Prompt for simple interactive chatbot.",
)


@pixie.app
async def example_chatbot():
    """A simple chatbot using Pydantic-AI agent with GPT-4o-mini.

    An OpenAI API key environment variable *(`OPENAI_API_KEY`)* is required to run this example.
    """
    agent = Agent(
        name="Simple chatbot",
        instructions=prompt.compile(),
        model="gpt-4o-mini",
    )

    yield "How can I help you today?"
    messages = []
    while True:
        user_msg = yield pixie.InputRequired(str)
        response = await agent.run(user_msg, message_history=messages)
        messages = response.all_messages()
        yield response.output
