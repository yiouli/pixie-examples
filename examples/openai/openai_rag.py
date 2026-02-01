"""
OpenAI RAG Agent (Retrieval Augmented Generation)

This example demonstrates building an agentic RAG system using the OpenAI Responses API
that can:
1. Decide when to use retrieval vs. respond directly
2. Grade retrieved documents for relevance
3. Rewrite questions if documents aren't relevant
4. Generate answers based on retrieved context

This is a direct port of the LangGraph RAG example, using native OpenAI API
instead of LangGraph/LangChain framework (except for vectorstore utilities).
"""

import json
from openai.types.responses import ResponseInputItemParam
from openai.types.responses.tool_param import ToolParam
from pydantic import BaseModel, Field
import langfuse.openai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

import pixie.sdk as pixie
import requests
from bs4 import BeautifulSoup


# ============================================================================
# PROMPT DEFINITIONS
# ============================================================================


class GradeVariables(pixie.Variables):
    context: str
    question: str


class RewriteVariables(pixie.Variables):
    question: str


class GenerateVariables(pixie.Variables):
    question: str
    context: str


rag_grade_prompt = pixie.create_prompt(
    "rag_grade_documents",
    GradeVariables,
    description="Grades relevance of retrieved documents to user questions",
)
rag_rewrite_prompt = pixie.create_prompt(
    "rag_rewrite_question",
    RewriteVariables,
    description="Rewrites questions to improve semantic understanding",
)
rag_generate_prompt = pixie.create_prompt(
    "rag_generate_answer",
    GenerateVariables,
    description="Generates concise answers from retrieved context",
)


# ============================================================================
# DOCUMENT LOADING & VECTORSTORE
# ============================================================================


def load_web_page(url: str) -> list[Document]:
    """Simple web page loader using requests and BeautifulSoup.

    Replaces langchain_community.document_loaders.WebBaseLoader
    to avoid the langchain-community dependency.
    """
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")

    # Extract text from the page
    text = soup.get_text(separator="\n", strip=True)

    return [Document(page_content=text, metadata={"source": url})]


def setup_vectorstore():
    """Setup vectorstore with documents from Lilian Weng's blog."""
    print("Loading documents from web...")

    urls = [
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]

    docs = [load_web_page(url) for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)

    print("Creating vectorstore...")
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits, embedding=OpenAIEmbeddings()
    )

    return vectorstore.as_retriever()


# ============================================================================
# SCHEMAS
# ============================================================================


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


# Tool definition for the retrieval
RETRIEVE_TOOL: ToolParam = {
    "type": "function",
    "name": "retrieve_blog_posts",
    "description": (
        "Search and return information about Lilian Weng blog posts on AI topics like reward hacking, "
        "hallucination, and diffusion video."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant blog post content",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "strict": True,
}


# ============================================================================
# RAG AGENT CLASS
# ============================================================================


class OpenAIRAGAgent:
    """Agentic RAG system using OpenAI Responses API."""

    def __init__(self, retriever, model: str = "gpt-4o-mini"):
        self.retriever = retriever
        self.model = model
        self.client = langfuse.openai.AsyncOpenAI()  # type: ignore
        self.message_history: list[ResponseInputItemParam] = []

    def retrieve(self, query: str) -> str:
        """Search and return information about Lilian Weng blog posts."""
        docs = self.retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])

    async def generate_query_or_respond(
        self, messages: list[ResponseInputItemParam]
    ) -> tuple[str | None, str | None]:
        """
        Call the model to decide whether to retrieve or respond directly.

        Returns:
            Tuple of (response_text, tool_query):
            - If responding directly: (response_text, None)
            - If tool call: (None, tool_query)
        """
        response = await self.client.responses.create(
            model=self.model,
            input=messages,
            tools=[RETRIEVE_TOOL],
        )

        # Check if the model wants to use a tool
        for item in response.output:
            if item.type == "function_call" and item.name == "retrieve_blog_posts":
                args = json.loads(item.arguments)
                return None, args["query"]

        # Otherwise, return the text response
        for item in response.output:
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        return content.text, None

        return None, None

    async def grade_documents(self, question: str, context: str) -> bool:
        """
        Determine whether the retrieved documents are relevant to the question.

        Returns:
            True if documents are relevant, False otherwise
        """
        prompt = rag_grade_prompt.compile(
            GradeVariables(question=question, context=context)
        )

        # Add instruction for JSON format (required when using json_object format)
        prompt_with_json = f"{prompt}\n\nProvide your response in JSON format with a 'binary_score' field."

        response = await self.client.responses.create(
            model="gpt-4o",
            input=[{"role": "user", "content": prompt_with_json}],
            text={"format": {"type": "json_object"}},
        )

        # Parse the response
        for item in response.output:
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        try:
                            result = json.loads(content.text)
                            return result.get("binary_score", "").lower() == "yes"
                        except json.JSONDecodeError:
                            # If parsing fails, check for "yes" in text
                            return "yes" in content.text.lower()

        return False

    async def rewrite_question(self, question: str) -> str:
        """Rewrite the original user question for better retrieval."""
        prompt = rag_rewrite_prompt.compile(RewriteVariables(question=question))

        response = await self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
        )

        for item in response.output:
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        return content.text

        return question  # Return original if rewrite fails

    async def generate_answer(self, question: str, context: str) -> str:
        """Generate an answer based on the retrieved context."""
        prompt = rag_generate_prompt.compile(
            GenerateVariables(question=question, context=context)
        )

        response = await self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
        )

        for item in response.output:
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        return content.text

        return "Unable to generate answer."

    async def run(
        self,
        question: str,
        max_rewrites: int = 2,
    ) -> str:
        """
        Run the RAG agent workflow.

        The workflow:
        1. Decide whether to retrieve or respond directly
        2. If retrieving, grade documents for relevance
        3. If documents aren't relevant, rewrite question and retry
        4. Generate answer based on relevant context

        Args:
            question: The user's question
            max_rewrites: Maximum number of question rewrites to attempt

        Returns:
            The generated answer
        """
        current_question = question
        rewrites = 0

        while True:
            print(f"Processing question: {current_question}")

            # Step 1: Decide whether to retrieve or respond directly
            user_msg: ResponseInputItemParam = {
                "role": "user",
                "content": current_question,
            }
            messages: list[ResponseInputItemParam] = [*self.message_history, user_msg]
            direct_response, tool_query = await self.generate_query_or_respond(messages)

            if direct_response is not None:
                # Model decided to respond directly without retrieval
                self.message_history.append(user_msg)
                self.message_history.append(
                    {"role": "assistant", "content": direct_response}
                )
                print("Model responded directly without retrieval")
                return direct_response

            if tool_query is None:
                # Unexpected state - no response or tool call
                return "Unable to process the question."

            # Step 2: Retrieve documents
            print(f"Retrieving documents for query: {tool_query}")
            context = self.retrieve(tool_query)

            # Step 3: Grade documents for relevance
            print("Grading documents for relevance...")
            is_relevant = await self.grade_documents(question, context)

            if is_relevant:
                # Step 4: Generate answer from relevant context
                print("Documents are relevant. Generating answer...")
                answer = await self.generate_answer(question, context)
                self.message_history.append(user_msg)
                self.message_history.append({"role": "assistant", "content": answer})
                return answer
            else:
                # Documents not relevant - try rewriting
                if rewrites >= max_rewrites:
                    print(
                        f"Max rewrites ({max_rewrites}) reached. "
                        "Generating answer with available context..."
                    )
                    answer = await self.generate_answer(question, context)
                    self.message_history.append(user_msg)
                    self.message_history.append(
                        {"role": "assistant", "content": answer}
                    )
                    return answer

                print("Documents not relevant. Rewriting question...")
                current_question = await self.rewrite_question(current_question)
                rewrites += 1
                print(f"Rewritten question: {current_question}")


# ============================================================================
# PIXIE APP
# ============================================================================


@pixie.app
async def openai_rag_agent() -> pixie.PixieGenerator[str, str]:
    """Interactive agentic RAG system for questions about Lilian Weng's blog posts.

    Uses OpenAI Responses API directly (without LangGraph/LangChain agent framework).

    The agent:
    1. Decides whether to retrieve or respond directly
    2. Grades retrieved documents for relevance
    3. Rewrites questions if needed
    4. Generates answers based on context

    Yields:
        AI-generated responses

    Receives:
        User questions about the blog content via InputRequired
    """
    # Setup retriever (this will take a moment on first run)
    retriever = setup_vectorstore()

    # Create the RAG agent
    agent = OpenAIRAGAgent(retriever, model="gpt-4o-mini")

    yield "Hello! I can answer questions about Lilian Weng's blog posts on AI topics."
    yield "Ask me anything about reward hacking, hallucination, or diffusion video!"
    yield "(Type 'exit' to quit)"

    while True:
        # Get user question
        user_question = yield pixie.InputRequired(str)

        # Check for exit
        if user_question.lower() in {"exit", "quit", "bye"}:
            yield "Goodbye! Thanks for chatting!"
            break

        # Run the RAG workflow
        answer = await agent.run(user_question)
        yield answer
