"""
Financial Research Agent - Multi-Step Workflow - Pixie Integration

This example demonstrates a structured financial research workflow with multiple
specialized agents working together to produce comprehensive financial analysis reports.

Pattern: Multi-Step Workflow with Multi-Agent
Original: https://github.com/openai/openai-agents-python/tree/main/examples/financial_research_agent

Architecture:
1. Planner Agent - Creates search plan
2. Search Agent - Performs web searches
3. Specialist Analysts - Financial fundamentals & risk analysis
4. Writer Agent - Synthesizes final report
5. Verifier Agent - Quality checks the report
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from pydantic import BaseModel
from agents import Agent, Runner, RunResult, RunResultStreaming, WebSearchTool
import pixie


# ============================================================================
# AGENT DEFINITIONS
# ============================================================================


# --- Planner Agent ---
class FinancialSearchItem(BaseModel):
    """A single search to perform"""

    reason: str
    """Your reasoning for why this search is relevant."""

    query: str
    """The search term to feed into a web (or file) search."""


class FinancialSearchPlan(BaseModel):
    """Plan of searches to perform"""

    searches: list[FinancialSearchItem]
    """A list of searches to perform."""


PLANNER_PROMPT = (
    "You are a financial research planner. Given a request for financial analysis, "
    "produce a set of web searches to gather the context needed. Aim for recent "
    "headlines, earnings calls or 10‑K snippets, analyst commentary, and industry background. "
    "Output between 5 and 15 search terms to query for."
)

planner_agent = Agent(
    name="FinancialPlannerAgent",
    instructions=PLANNER_PROMPT,
    model="o3-mini",
    output_type=FinancialSearchPlan,
)


# --- Search Agent ---
SEARCH_INSTRUCTIONS = (
    "You are a research assistant specializing in financial topics. "
    "Given a search term, use web search to retrieve up‑to‑date context and "
    "produce a short summary of at most 300 words. Focus on key numbers, events, "
    "or quotes that will be useful to a financial analyst."
)

search_agent = Agent(
    name="FinancialSearchAgent",
    model="gpt-5.2",
    instructions=SEARCH_INSTRUCTIONS,
    tools=[WebSearchTool()],
)


# --- Analyst Agents ---
class AnalysisSummary(BaseModel):
    """Analysis output from specialist agents"""

    summary: str
    """Short text summary for this aspect of the analysis."""


FINANCIALS_PROMPT = (
    "You are a financial analyst focused on company fundamentals such as revenue, "
    "profit, margins and growth trajectory. Given a collection of web (and optional file) "
    "search results about a company, write a concise analysis of its recent financial "
    "performance. Pull out key metrics or quotes. Keep it under 2 paragraphs."
)

financials_agent = Agent(
    name="FundamentalsAnalystAgent",
    instructions=FINANCIALS_PROMPT,
    output_type=AnalysisSummary,
)

RISK_PROMPT = (
    "You are a risk analyst looking for potential red flags in a company's outlook. "
    "Given background research, produce a short analysis of risks such as competitive threats, "
    "regulatory issues, supply chain problems, or slowing growth. Keep it under 2 paragraphs."
)

risk_agent = Agent(
    name="RiskAnalystAgent",
    instructions=RISK_PROMPT,
    output_type=AnalysisSummary,
)


# --- Writer Agent ---
class FinancialReportData(BaseModel):
    """Final report output"""

    short_summary: str
    """A short 2‑3 sentence executive summary."""

    markdown_report: str
    """The full markdown report."""

    follow_up_questions: list[str]
    """Suggested follow‑up questions for further research."""


WRITER_PROMPT = (
    "You are a senior financial analyst. You will be provided with the original query and "
    "a set of raw search summaries. Your task is to synthesize these into a long‑form markdown "
    "report (at least several paragraphs) including a short executive summary and follow‑up "
    "questions. If needed, you can call the available analysis tools (e.g. fundamentals_analysis, "
    "risk_analysis) to get short specialist write‑ups to incorporate."
)

writer_agent = Agent(
    name="FinancialWriterAgent",
    instructions=WRITER_PROMPT,
    model="gpt-5.2",
    output_type=FinancialReportData,
)


# --- Verifier Agent ---
class VerificationResult(BaseModel):
    """Verification outcome"""

    verified: bool
    """Whether the report seems coherent and plausible."""

    issues: str
    """If not verified, describe the main issues or concerns."""


VERIFIER_PROMPT = (
    "You are a meticulous auditor. You have been handed a financial analysis report. "
    "Your job is to verify the report is internally consistent, clearly sourced, and makes "
    "no unsupported claims. Point out any issues or uncertainties."
)

verifier_agent = Agent(
    name="VerificationAgent",
    instructions=VERIFIER_PROMPT,
    model="gpt-5.2",
    output_type=VerificationResult,
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


async def _summary_extractor(run_result: RunResult | RunResultStreaming) -> str:
    """Custom output extractor for sub‑agents that return an AnalysisSummary."""
    return str(run_result.final_output.summary)


async def _plan_searches(query: str) -> FinancialSearchPlan:
    """Create a search plan for the given query"""
    result = await Runner.run(planner_agent, f"Query: {query}")
    return result.final_output_as(FinancialSearchPlan)


async def _perform_search(item: FinancialSearchItem) -> str | None:
    """Perform a single search"""
    input_data = f"Search term: {item.query}\nReason: {item.reason}"
    try:
        result = await Runner.run(search_agent, input_data)
        return str(result.final_output)
    except Exception:
        return None


async def _perform_searches(search_plan: FinancialSearchPlan) -> Sequence[str]:
    """Perform all searches in parallel"""
    tasks = [
        asyncio.create_task(_perform_search(item)) for item in search_plan.searches
    ]
    results: list[str] = []

    for task in asyncio.as_completed(tasks):
        result = await task
        if result is not None:
            results.append(result)

    return results


async def _write_report(
    query: str, search_results: Sequence[str]
) -> FinancialReportData:
    """Write the final report using specialist tools"""
    # Expose the specialist analysts as tools
    fundamentals_tool = financials_agent.as_tool(
        tool_name="fundamentals_analysis",
        tool_description="Use to get a short write‑up of key financial metrics",
        custom_output_extractor=_summary_extractor,
    )
    risk_tool = risk_agent.as_tool(
        tool_name="risk_analysis",
        tool_description="Use to get a short write‑up of potential red flags",
        custom_output_extractor=_summary_extractor,
    )

    writer_with_tools = writer_agent.clone(tools=[fundamentals_tool, risk_tool])
    input_data = f"Original query: {query}\nSummarized search results: {search_results}"

    result = await Runner.run(writer_with_tools, input_data)
    return result.final_output_as(FinancialReportData)


async def _verify_report(report: FinancialReportData) -> VerificationResult:
    """Verify report quality"""
    result = await Runner.run(verifier_agent, report.markdown_report)
    return result.final_output_as(VerificationResult)


# ============================================================================
# PIXIE APP
# ============================================================================


@pixie.app
async def openai_agents_financial_research(
    query: str,
) -> pixie.PixieGenerator[str, None]:
    """
    Comprehensive financial research agent with multi-step workflow.

    This agent orchestrates a full research workflow:
    1. Plans searches based on the query
    2. Executes web searches in parallel
    3. Analyzes fundamentals and risks using specialist agents
    4. Synthesizes a comprehensive markdown report
    5. Verifies the report for quality and consistency

    Args:
        query: Financial research question or company to analyze

    Yields:
        Progress updates and the final comprehensive report

    Example queries:
        - "Write up an analysis of Apple Inc.'s most recent quarter."
        - "Analyze Tesla's financial performance and growth prospects"
        - "What are the key risks facing Microsoft in 2026?"
    """
    yield f"🔍 Starting financial research for: {query}\n"

    # Step 1: Planning
    yield "📋 Planning searches..."
    search_plan = await _plan_searches(query)
    num_searches = len(search_plan.searches)
    yield f"✓ Will perform {num_searches} searches\n"

    # Show search plan
    yield "Search plan:"
    for i, item in enumerate(search_plan.searches[:5], 1):  # Show first 5
        yield f"  {i}. {item.query} - {item.reason}"
    if num_searches > 5:
        yield f"  ... and {num_searches - 5} more"
    yield ""

    # Step 2: Searching
    yield f"🔎 Executing {num_searches} searches in parallel..."
    search_results = await _perform_searches(search_plan)
    yield f"✓ Completed {len(search_results)} successful searches\n"

    # Step 3: Writing report
    yield "✍️ Analyzing data and writing comprehensive report..."
    yield "(This may take a minute as specialist analysts are consulted...)"
    report = await _write_report(query, search_results)
    yield "✓ Report complete\n"

    # Step 4: Verification
    yield "🔍 Verifying report quality..."
    verification = await _verify_report(report)
    if verification.verified:
        yield "✓ Report verified\n"
    else:
        yield f"⚠️ Verification issues found: {verification.issues}\n"

    # Output summary
    yield "=" * 60
    yield "EXECUTIVE SUMMARY"
    yield "=" * 60
    yield report.short_summary
    yield ""

    # Full report
    yield "=" * 60
    yield "FULL REPORT"
    yield "=" * 60
    yield report.markdown_report
    yield ""

    # Follow-up questions
    yield "=" * 60
    yield "SUGGESTED FOLLOW-UP QUESTIONS"
    yield "=" * 60
    for i, question in enumerate(report.follow_up_questions, 1):
        yield f"{i}. {question}"

    yield ""
    yield "✓ Financial research complete!"
