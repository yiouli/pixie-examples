from pydantic_ai import Agent

import pixie

agent = Agent(
    name="Simple chatbot",
    instructions="You are a helpful assistant.",
    model="gpt-4o-mini",
)


@pixie.pixie_app
async def example_chatbot(_):
    """A simple chatbot using Pydantic-AI agent with GPT-4o-mini.

    An OpenAI API key environment variable *(`OPENAI_API_KEY`)* is required to run this example.
    """

    yield "How can I help you today?"
    messages = []
    while True:
        user_msg = yield pixie.UserInputRequirement(str)
        response = await agent.run(user_msg, message_history=messages)
        messages = response.all_messages()
        yield response.output
