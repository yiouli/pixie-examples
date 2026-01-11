"""
Customer Support with Handoffs (State Machine)

This example demonstrates the state machine pattern where an agent's behavior changes
as it moves through different states of a workflow.

Based on: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support
"""

from typing import Literal, NotRequired
from langchain.agents import create_agent, AgentState
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from typing import Callable
from pixie import pixie_app, PixieGenerator, UserInputRequirement


# Define the possible workflow steps
SupportStep = Literal["warranty_collector", "issue_classifier", "resolution_specialist"]


class SupportState(AgentState):
    """State for customer support workflow."""

    current_step: NotRequired[SupportStep]
    warranty_status: NotRequired[Literal["in_warranty", "out_of_warranty"]]
    issue_type: NotRequired[Literal["hardware", "software"]]


# Define tools that manage workflow state
@tool
def record_warranty_status(
    status: Literal["in_warranty", "out_of_warranty"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record the customer's warranty status and transition to issue classification."""
    return Command(
        update={
            "messages": [
                {
                    "role": "tool",
                    "content": f"Warranty status recorded as: {status}",
                    "tool_call_id": runtime.tool_call_id,
                }
            ],
            "warranty_status": status,
            "current_step": "issue_classifier",
        }
    )


@tool
def record_issue_type(
    issue_type: Literal["hardware", "software"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record the type of issue and transition to resolution specialist."""
    return Command(
        update={
            "messages": [
                {
                    "role": "tool",
                    "content": f"Issue type recorded as: {issue_type}",
                    "tool_call_id": runtime.tool_call_id,
                }
            ],
            "issue_type": issue_type,
            "current_step": "resolution_specialist",
        }
    )


@tool
def escalate_to_human(reason: str) -> str:
    """Escalate the case to a human support specialist."""
    return f"Escalating to human support. Reason: {reason}"


@tool
def provide_solution(solution: str) -> str:
    """Provide a solution to the customer's issue."""
    return f"Solution provided: {solution}"


# Step prompts
WARRANTY_COLLECTOR_PROMPT = """You are a customer support agent collecting warranty information.

CURRENT STAGE: Warranty Verification

Ask the customer if their device is under warranty. Once you have this information,
use the record_warranty_status tool to record it and move to the next step.

Be polite and professional."""

ISSUE_CLASSIFIER_PROMPT = """You are a customer support agent classifying technical issues.

CURRENT STAGE: Issue Classification
CUSTOMER INFO: Warranty status is {warranty_status}

Ask the customer to describe their issue, then determine if it's:
- HARDWARE: Physical problems (cracked screen, battery, ports, buttons)
- SOFTWARE: App crashes, performance, settings, updates

Use record_issue_type to record the classification and move to resolution."""

RESOLUTION_SPECIALIST_PROMPT = """You are a customer support agent helping with device issues.

CURRENT STAGE: Resolution
CUSTOMER INFO: Warranty status is {warranty_status}, issue type is {issue_type}

At this step, you need to:
1. For SOFTWARE issues: provide troubleshooting steps using provide_solution
2. For HARDWARE issues:
   - If IN WARRANTY: explain warranty repair process using provide_solution
   - If OUT OF WARRANTY: escalate_to_human for paid repair options

Be specific and helpful in your solutions."""

# Step configuration
STEP_CONFIG = {
    "warranty_collector": {
        "prompt": WARRANTY_COLLECTOR_PROMPT,
        "tools": [record_warranty_status],
        "requires": [],
    },
    "issue_classifier": {
        "prompt": ISSUE_CLASSIFIER_PROMPT,
        "tools": [record_issue_type],
        "requires": ["warranty_status"],
    },
    "resolution_specialist": {
        "prompt": RESOLUTION_SPECIALIST_PROMPT,
        "tools": [provide_solution, escalate_to_human],
        "requires": ["warranty_status", "issue_type"],
    },
}


# Create step-based middleware
@wrap_model_call
def apply_step_config(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Configure agent behavior based on the current step."""
    # Get current step (defaults to warranty_collector for first interaction)
    current_step = request.state.get("current_step", "warranty_collector")

    # Look up step configuration
    stage_config = STEP_CONFIG[current_step]

    # Validate required state exists
    for key in stage_config["requires"]:
        if request.state.get(key) is None:
            raise ValueError(f"{key} must be set before reaching {current_step}")

    # Format prompt with state values
    # Note: In a production implementation, you would inject the formatted prompt
    # and tools into the request. For simplicity, we'll let the handler process
    # the request and handle tool selection based on state.
    _ = stage_config["prompt"].format(**request.state)

    # The middleware pattern here would need deeper integration with LangChain's
    # internal APIs. For now, we pass through to the handler.
    return handler(request)


@pixie_app
async def customer_support(_: None) -> PixieGenerator[str, str]:
    """Customer support agent with state machine workflow.

    The agent progresses through three stages:
    1. Warranty verification
    2. Issue classification (hardware/software)
    3. Resolution (solution or escalation)

    Args:
        _: No initial input required

    Yields:
        AI responses guiding the support workflow
    """
    # Initialize model
    model = init_chat_model("gpt-4o-mini", temperature=0)

    # Collect all tools
    all_tools = [
        record_warranty_status,
        record_issue_type,
        provide_solution,
        escalate_to_human,
    ]

    # Create agent with step-based configuration
    agent = create_agent(
        model,
        tools=all_tools,
        state_schema=SupportState,
        middleware=[apply_step_config],
        checkpointer=InMemorySaver(),
    )

    # Send welcome message
    yield "Welcome to customer support! I'm here to help with your device issue."

    # Initialize conversation
    thread_id = "support_thread"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        # Get user input
        user_message = yield UserInputRequirement(str)

        # Check for exit
        if user_message.lower() in {"exit", "quit", "bye"}:
            yield "Thank you for contacting support. Have a great day!"
            break

        # Process with agent
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]}, config  # type: ignore
        )

        # Yield the agent's response
        yield result["messages"][-1].content

        # Check if we've reached a resolution
        current_state = result
        if current_state.get("current_step") == "resolution_specialist" and any(
            msg.get("role") == "tool"
            and msg.get("name") in ["provide_solution", "escalate_to_human"]
            for msg in result.get("messages", [])
        ):
            yield "Is there anything else I can help you with? (Type 'exit' to end)"
