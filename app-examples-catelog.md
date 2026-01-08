# Selected AI Framework Examples for Pixie Integration

This document contains the curated selection of example applications/agents from major AI frameworks for demonstrating Pixie integration. Each framework has 3-5 examples selected based on specific criteria: quickstart, multi-step workflow, multi-agent, multi-turn chatbot, and graph/state-machine patterns.

---

## Pydantic AI

### 1. Pydantic Model (Quickstart)

**Description:** Basic agent with structured output validation using output_type parameter
**Reason:** Perfect bread & butter quickstart showing core Pydantic AI concepts with type-safe structured outputs
**Links:**

- [Documentation](https://ai.pydantic.dev/examples/pydantic-model/)
- [Source Code](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_model.py)

### 2. Bank Support (Multi-Turn Chatbot)

**Description:** Flagship example demonstrating dependency injection, dynamic instructions, and tools
**Reason:** Excellent multi-turn conversational example showing real-world chat application patterns with context management
**Links:**

- [Documentation](https://ai.pydantic.dev/examples/bank-support/)
- [Source Code](https://github.com/pydantic/pydantic-ai/blob/main/examples/bank_support.py)

### 3. Flight Booking (Multi-Agent)

**Description:** Multi-agent delegation with human-in-the-loop confirmation and usage limits
**Reason:** Demonstrates multi-agent patterns with delegation and HITL, a common production pattern
**Links:**

- [Documentation](https://ai.pydantic.dev/examples/flight-booking/)
- [Source Code](https://github.com/pydantic/pydantic-ai/blob/main/examples/flight_booking.py)

### 4. Question Graph (Graph/State-Machine)

**Description:** Graph-based workflow using Pydantic Graph with state persistence
**Reason:** Shows graph-based state machine pattern using Pydantic's Graph feature
**Links:**

- [Documentation](https://ai.pydantic.dev/examples/question-graph/)
- [Source Code](https://github.com/pydantic/pydantic-ai/blob/main/examples/question_graph.py)

### 5. SQL Generation (Multi-Step Workflow)

**Description:** Text-to-SQL with output validation via EXPLAIN queries and ModelRetry correction
**Reason:** Demonstrates multi-step workflow with validation and error correction loops
**Links:**

- [Documentation](https://ai.pydantic.dev/examples/sql-gen/)
- [Source Code](https://github.com/pydantic/pydantic-ai/blob/main/examples/sql_gen.py)

---

## OpenAI Agents SDK

### 1. Deterministic Workflows (Quickstart & Multi-Step)

**Description:** Sequential steps where each agent's output feeds into the next (story generation pipeline)
**Reason:** Clean quickstart showing basic sequential workflow pattern fundamental to agent development
**Links:**

- [Source Code](https://github.com/openai/openai-agents-python/tree/main/examples/agent_patterns/deterministic_workflows)

### 2. Customer Service (Multi-Agent & Multi-Turn)

**Description:** Airline support with triage, FAQ, and seat booking agents with handoffs
**Reason:** Comprehensive example showing multi-agent coordination with handoffs and conversational patterns
**Links:**

- [Source Code](https://github.com/openai/openai-agents-python/tree/main/examples/application_examples/customer_service)

### 3. Routing/Handoffs (Graph/State-Machine)

**Description:** Triage agent routing to specialists based on language detection
**Reason:** Demonstrates routing and handoff patterns which are essentially state machine transitions
**Links:**

- [Source Code](https://github.com/openai/openai-agents-python/tree/main/examples/agent_patterns/routing_handoffs)

### 4. Financial Research Agent (Multi-Step Workflow)

**Description:** Manager orchestrating planning, analysis, writing, and verification phases
**Reason:** Shows sophisticated multi-phase workflow with orchestration and verification
**Links:**

- [Source Code](https://github.com/openai/openai-agents-python/tree/main/examples/application_examples/financial_research_agent)

---

## Crew AI

### 1. Collaboration (Quickstart)

**Description:** Agent collaboration patterns and delegation
**Reason:** Core quickstart demonstrating fundamental Crew AI collaboration concepts
**Links:**

- [Source Code](https://github.com/crewAIInc/crewAI-examples/tree/main/quickstarts/collaboration)

### 2. Self Evaluation Loop Flow (Multi-Step Workflow & Graph)

**Description:** Iterative content improvement with feedback loops
**Reason:** Shows both multi-step workflow with evaluation loops and state-based flow control
**Links:**

- [Source Code](https://github.com/crewAIInc/crewAI-examples/tree/main/self_evaluation_loop_flow)

### 3. Lead Score Flow (Multi-Agent & Multi-Turn)

**Description:** Lead qualification with human-in-the-loop review
**Reason:** Demonstrates multi-agent crew working together with HITL interaction pattern
**Links:**

- [Source Code](https://github.com/crewAIInc/crewAI-examples/tree/main/lead_score_flow)

### 4. Marketing Strategy (Multi-Agent)

**Description:** Multi-agent crew for marketing strategy development
**Reason:** Classic Crew AI example showing multiple agents with different roles collaborating
**Links:**

- [Source Code](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/marketing_strategy)

### 5. Content Creator Flow (Multi-Step Workflow)

**Description:** Multi-crew content generation for blogs, LinkedIn, and research reports
**Reason:** Demonstrates complex multi-step flow orchestrating multiple crews
**Links:**

- [Source Code](https://github.com/crewAIInc/crewAI-examples/tree/main/content_creator_flow)

---

## Claude Agents SDK

### 1. Hello World (Quickstart)

**Description:** Basic SDK fundamentals with query(), streaming, and configuration
**Reason:** Essential quickstart for understanding Claude SDK basics
**Links:**

- [Documentation](https://platform.claude.com/docs/en/agent-sdk/examples/hello-world)
- [Source Code](https://github.com/anthropics/claude-agent-sdk-python/tree/main/examples/hello_world)

### 2. Email Agent (Multi-Turn & Multi-Step)

**Description:** IMAP email assistant with context gathering and tool use
**Reason:** Shows multi-turn conversation with context management and multi-step email processing workflow
**Links:**

- [Blog Post](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)
- [Source Code](https://github.com/anthropics/claude-agent-sdk-python/tree/main/examples/email_agent)

### 3. Research Agent (Multi-Agent)

**Description:** Multi-agent system with parallel subagents and SDK hooks
**Reason:** Demonstrates multi-agent orchestration with parallel execution of subagents
**Links:**

- [Documentation](https://platform.claude.com/docs/en/agent-sdk/examples/research-agent)
- [Source Code](https://github.com/anthropics/claude-agent-sdk-python/tree/main/examples/research_agent)

### 4. Customer Support Agent (Multi-Turn Chatbot)

**Description:** AI-assisted support with knowledge base access
**Reason:** Classic chatbot example with conversation management and external knowledge
**Links:**

- [Source Code](https://github.com/anthropics/anthropic-quickstarts/tree/main/customer-support-agent)

### 5. Basic Workflows (Multi-Step & Graph)

**Description:** Prompt-chaining, parallelization, and routing patterns
**Reason:** Demonstrates various workflow patterns including routing which creates state machine behavior
**Links:**

- [Source Code](https://github.com/anthropics/anthropic-cookbook/tree/main/patterns/workflows)

---

## LangChain

### 1. RAG Agent (Quickstart)

**Description:** Q&A over unstructured text with semantic search
**Reason:** Perfect quickstart showing fundamental RAG agent pattern with LangChain
**Links:**

- [Documentation](https://python.langchain.com/docs/tutorials/agents/)

### 2. Supervisor Agent (Multi-Agent)

**Description:** Multi-agent with calendar and email sub-agents
**Reason:** Clear multi-agent example with supervisor coordination pattern
**Links:**

- [Documentation](https://python.langchain.com/docs/tutorials/agents/#supervisor-agent)

### 3. Handoffs (Graph/State-Machine)

**Description:** Customer support state machine workflow
**Reason:** Excellent example of state machine pattern with handoff transitions
**Links:**

- [Documentation](https://python.langchain.com/docs/concepts/multi_agent/#handoffs)

### 4. SQL Agent (Multi-Turn & Multi-Step)

**Description:** Database interaction with schema queries and human-in-the-loop
**Reason:** Shows multi-turn conversation with HITL and multi-step database operations
**Links:**

- [Documentation](https://python.langchain.com/docs/tutorials/sql_agent/)

### 5. Email Assistant (Multi-Turn Chatbot)

**Description:** Email assistant with evaluation, HITL, and memory (4-part tutorial)
**Reason:** Comprehensive chatbot tutorial with memory and evaluation patterns
**Links:**

- [Source Code](https://github.com/langchain-ai/agents-from-scratch)

---

## LangGraph

### 1. Agent Supervisor (Quickstart & Multi-Agent)

**Description:** LLM orchestrating researcher and chart_generator workers
**Reason:** Clean quickstart showing basic LangGraph concepts with multi-agent supervision
**Links:**

- [Documentation](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/)

### 2. Customer Support Bot (Multi-Turn Chatbot & Graph)

**Description:** Airline support with flight, hotel, and car rental bookings
**Reason:** Complete chatbot application using graph-based state management
**Links:**

- [Documentation](https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/)

### 3. Plan-and-Execute (Multi-Step Workflow)

**Description:** Multi-step planning with sequential execution and revision
**Reason:** Classic multi-step workflow pattern with planning and execution phases
**Links:**

- [Documentation](https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/)

### 4. Hierarchical Teams (Multi-Agent & Graph)

**Description:** Top-level supervisor coordinating mid-level supervisors and workers
**Reason:** Advanced multi-agent hierarchy demonstrating complex graph-based coordination
**Links:**

- [Documentation](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/hierarchical_agent_teams/)

### 5. Adaptive RAG (Graph/State-Machine)

**Description:** Query analysis with active/self-corrective retrieval routing
**Reason:** Shows sophisticated state machine with dynamic routing based on query analysis
**Links:**

- [Documentation](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/)

---

## Google ADK

### 1. Multi-Tool Agent Quickstart (Quickstart)

**Description:** Weather and time tools with Dev UI, terminal, and API server
**Reason:** Official quickstart demonstrating core ADK concepts with tools
**Links:**

- [Documentation](https://google.github.io/adk-docs/guides/multi-tool-agent/)

### 2. Agent Team (Weather Bot) (Multi-Agent & Multi-Turn)

**Description:** Multi-agent delegation, session state, callbacks, and LiteLLM
**Reason:** Shows multi-agent team coordination with conversational state management
**Links:**

- [Documentation](https://google.github.io/adk-docs/guides/agent-team/)

### 3. Workflow Agents (Multi-Step & Graph)

**Description:** SequentialAgent, ParallelAgent, LoopAgent for deterministic control
**Reason:** Demonstrates various workflow patterns including sequential and loop-based state machines
**Links:**

- [Consuming Documentation](https://google.github.io/adk-docs/guides/consuming-a2a-agents/)
- [Exposing Documentation](https://google.github.io/adk-docs/guides/exposing-a2a-agents/)

### 4. Customer Service (Multi-Turn Chatbot & Multi-Step)

**Description:** Async tools, live streaming, multimodal (home improvement domain)
**Reason:** Production-ready customer service chatbot with async multi-step operations (excluding streaming for Pixie demo)
**Links:**

- [Source Code](https://github.com/google/adk-samples/tree/main/python/agents/customer_service)

### 5. Financial Advisor (Multi-Turn Chatbot)

**Description:** Enterprise financial advisory agent with conversational interaction
**Reason:** Domain-specific chatbot showing multi-turn conversation management
**Links:**

- [Source Code](https://github.com/google/adk-samples/tree/main/python/agents/financial_advisor)

---

## Summary

This selection provides comprehensive coverage of common agent patterns across all major frameworks:

- **Quickstart examples:** 7 frameworks covered
- **Multi-step workflows:** 7 frameworks covered
- **Multi-agent patterns:** 7 frameworks covered
- **Multi-turn chatbots:** 7 frameworks covered
- **Graph/state-machine patterns:** 6 frameworks covered

All selected examples:

- ✅ Run standalone without external infrastructure (no Docker, databases, etc.)
- ✅ Focus on core agent patterns without streaming or multimodal complexity
- ✅ Represent production-ready patterns developers actually use
- ✅ Have clear documentation and source code available
