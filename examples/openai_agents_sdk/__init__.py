"""
OpenAI Agents SDK Examples - Pixie Integration

This package contains Pixie-integrated versions of OpenAI Agents SDK examples.

Available handlers:
- llm_judge_story_generator: LLM-as-a-judge pattern for iterative story generation
- airline_customer_service: Multi-agent customer service with handoffs
- multilingual_routing: Language-based routing with streaming
- multilingual_routing_simple: Simplified language routing with initial message
- financial_research: Comprehensive financial research workflow

For usage details, see README.md
"""

from .llm_as_a_judge import llm_as_a_judge
from .customer_service import airline_customer_service
from .routing import multilingual_routing, multilingual_routing_simple
from .financial_research_agent import financial_research

__all__ = [
    "llm_as_a_judge",
    "airline_customer_service",
    "multilingual_routing",
    "multilingual_routing_simple",
    "financial_research",
]
