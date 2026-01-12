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
import pixie


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


# System prompts for specialized agents
CALENDAR_AGENT_PROMPT = (
    "You are a calendar scheduling assistant. "
    "Parse natural language scheduling requests (e.g., 'next Tuesday at 2pm') "
    "into proper ISO datetime formats. "
    "Use get_available_time_slots to check availability when needed. "
    "Use create_calendar_event to schedule events. "
    "Always confirm what was scheduled in your final response."
)

EMAIL_AGENT_PROMPT = (
    "You are an email assistant. "
    "Compose professional emails based on natural language requests. "
    "Extract recipient information and craft appropriate subject lines and body text. "
    "Use send_email to send the message. "
    "Always confirm what was sent in your final response."
)

SUPERVISOR_PROMPT = (
    "You are a helpful personal assistant. "
    "You can schedule calendar events and send emails. "
    "Break down user requests into appropriate tool calls and coordinate the results. "
    "When a request involves multiple actions, use multiple tools in sequence."
)


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
        system_prompt=CALENDAR_AGENT_PROMPT,
    )

    # Create email subagent
    email_agent = create_agent(
        model,
        tools=[send_email],
        system_prompt=EMAIL_AGENT_PROMPT,
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
        system_prompt=SUPERVISOR_PROMPT,
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
