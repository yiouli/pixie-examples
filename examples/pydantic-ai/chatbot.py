from types import NoneType
from pydantic_ai import Agent, ModelMessage, ModelRequest
from pixie import pixie_app, PixieGenerator, UserInputRequirement

chatbot = Agent(
    "openai:gpt-4o-mini",
    system_prompt=(
        "You are a friendly chatbot. Answer user questions concisely and helpfully."
    ),
)


@pixie_app
async def chat_with_ai(_: NoneType) -> PixieGenerator[str, str]:
    welcome = "Hello! How can I assist you today?"
    m = ModelRequest.user_text_prompt(welcome)
    history: list[ModelMessage] = [m]
    yield welcome
    while True:
        user_msg = yield UserInputRequirement(str)
        if user_msg.lower() in {"exit", "quit", "bye"}:
            yield "Goodbye! Have a great day!"
            break
        ai_response = await chatbot.run(user_msg, message_history=history)
        history.append(ModelRequest.user_text_prompt(user_msg))
        history.append(ai_response.response)
        yield ai_response.output
