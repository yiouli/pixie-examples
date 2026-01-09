from langfuse.openai import openai  # type: ignore
from pixie import pixie_app, PixieGenerator, UserInputRequirement

client = openai.AsyncClient()


@pixie_app
async def chat(_: None) -> PixieGenerator[str, str]:

    yield "How can I help you today?"
    messages = []
    while True:
        user_msg = yield UserInputRequirement(str)
        messages.append({"role": "user", "content": user_msg})
        response = await client.chat.completions.create(
            model="gpt-4o-mini", messages=messages
        )
        ai_response = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": ai_response})
        yield ai_response
