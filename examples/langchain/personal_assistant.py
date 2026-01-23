"""
Personal Assistant with Subagents (Multi-Agent)

This example demonstrates the supervisor pattern where a central supervisor agent
coordinates specialized worker agents (calendar and email agents).

Based on: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant
"""

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langfuse.langchain import CallbackHandler
import pixie.sdk as pixie


calendar_agent_prompt = pixie.create_prompt(
    "calendar_agent",
    description="Calendar scheduling assistant that parses natural language requests",
)
email_agent_prompt = pixie.create_prompt(
    "email_agent",
    description="Email assistant that composes and sends professional emails",
)
supervisor_agent_prompt = pixie.create_prompt(
    "supervisor_agent",
    description="Personal assistant coordinator that handles calendar and email tasks",
)

langfuse_handler = CallbackHandler()


# Define calendar tools (stubs for demonstration)
@tool
def create_calendar_event(
    title: str,
    start_time: str,  # ISO format: "2024-01-15T14:00:00"
    end_time: str,  # ISO format: "2024-01-15T15:00:00"
    attendees: list[str],  # email addresses
    location: str = "",
) -> str:
    """Create a calendar event. Requires exact ISO datetime format."""
    return f"Event created: {title} from {start_time} to {end_time} with {len(attendees)} attendees"


@tool
def get_available_time_slots(
    attendees: list[str], date: str, duration_minutes: int  # ISO format: "2024-01-15"
) -> list[str]:
    """Check calendar availability for given attendees on a specific date."""
    return ["09:00", "14:00", "16:00"]


@tool
def send_email(to: list[str], subject: str, body: str, cc: list[str] = []) -> str:
    """Send an email via email API. Requires properly formatted addresses."""
    return f"Email sent to {', '.join(to)} - Subject: {subject}"


@pixie.app
async def langchain_personal_assistant() -> pixie.PixieGenerator[str, str]:
    """Multi-agent personal assistant with calendar and email subagents.

    The supervisor coordinates specialized worker agents:
    - Calendar agent: handles scheduling and availability
    - Email agent: manages communication and drafts

    Yields:
        AI responses to user requests
    """
    # Initialize model
    model = init_chat_model("gpt-4o-mini", temperature=0)

    # Create calendar subagent
    calendar_agent = create_agent(
        model,
        tools=[create_calendar_event, get_available_time_slots],
        system_prompt=calendar_agent_prompt.compile(),
    )

    # Create email subagent
    email_agent = create_agent(
        model,
        tools=[send_email],
        system_prompt=email_agent_prompt.compile(),
    )

    # Wrap subagents as tools for the supervisor
    @tool
    def schedule_event(request: str) -> str:
        """Schedule calendar events using natural language.

        Use this when the user wants to create, modify, or check calendar appointments.
        Handles date/time parsing, availability checking, and event creation.
        """
        result = calendar_agent.invoke(
            {"messages": [{"role": "user", "content": request}]},
            config={"callbacks": [langfuse_handler]},
        )
        return result["messages"][-1].content

    @tool
    def manage_email(request: str) -> str:
        """Send emails using natural language.

        Use this when the user wants to send notifications, reminders, or any email
        communication. Handles recipient extraction, subject generation, and email composition.
        """
        result = email_agent.invoke(
            {"messages": [{"role": "user", "content": request}]},
            config={"callbacks": [langfuse_handler]},
        )
        return result["messages"][-1].content

    # Create supervisor agent with checkpointer for conversation memory
    supervisor_agent = create_agent(
        model,
        tools=[schedule_event, manage_email],
        system_prompt=supervisor_agent_prompt.compile(),
        checkpointer=InMemorySaver(),
    )

    # Send welcome message
    yield (
        "Hello! I'm your personal assistant. I can help you schedule events "
        "and send emails. What would you like me to do?"
    )

    # Initialize conversation
    thread_id = "personal_assistant_thread"
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [langfuse_handler]}

    while True:
        # Get user request
        user_request = yield pixie.InputRequired(str)

        # Check for exit
        if user_request.lower() in {"exit", "quit", "bye", "goodbye"}:
            yield "Goodbye! Let me know if you need anything else."
            break

        # Process request with supervisor
        result = supervisor_agent.invoke(
            {"messages": [{"role": "user", "content": user_request}]}, config  # type: ignore
        )

        # Yield the supervisor's response
        yield result["messages"][-1].content
