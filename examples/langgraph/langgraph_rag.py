"""
LangGraph RAG Agent (Retrieval Augmented Generation)

This example demonstrates building an agentic RAG system using LangGraph that can:
1. Decide when to use retrieval vs. respond directly
2. Grade retrieved documents for relevance
3. Rewrite questions if documents aren't relevant
4. Generate answers based on retrieved context

Based on: https://docs.langchain.com/oss/python/langgraph/agentic-rag
"""

from pydantic import BaseModel, Field
from typing import Literal, cast
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from langfuse.langchain import CallbackHandler
import pixie.sdk as pixie
import requests
from bs4 import BeautifulSoup


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

langfuse_handler = CallbackHandler()


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


def create_rag_graph(retriever, model):
    """Create a LangGraph-based RAG agent."""

    # Create retriever tool
    @tool
    def retrieve_blog_posts(query: str) -> str:
        """Search and return information about Lilian Weng blog posts."""
        docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])

    retriever_tool = retrieve_blog_posts

    # Node: Generate query or respond
    def generate_query_or_respond(state: MessagesState):
        """Call the model to generate a response or use retrieval tool."""
        response = model.bind_tools([retriever_tool]).invoke(
            state["messages"], config={"callbacks": [langfuse_handler]}
        )
        return {"messages": [response]}

    # Grade documents schema
    class GradeDocuments(BaseModel):
        """Grade documents using a binary score for relevance check."""

        binary_score: str = Field(
            description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
        )

    grader_model = init_chat_model("gpt-4o", temperature=0)

    # Conditional edge: Grade documents
    def grade_documents(
        state: MessagesState,
    ) -> Literal["generate_answer", "rewrite_question"]:
        """Determine whether the retrieved documents are relevant to the question."""
        question = cast(str, state["messages"][0].content)
        context = cast(str, state["messages"][-1].content)

        prompt = rag_grade_prompt.compile(
            GradeVariables(question=question, context=context)
        )
        response = grader_model.with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}],
            config={"callbacks": [langfuse_handler]},
        )

        score = response.binary_score  # type: ignore
        if score == "yes":
            return "generate_answer"
        else:
            return "rewrite_question"

    # Node: Rewrite question
    def rewrite_question(state: MessagesState):
        """Rewrite the original user question."""
        messages = state["messages"]
        question = cast(str, messages[0].content)
        prompt = rag_rewrite_prompt.compile(RewriteVariables(question=question))
        response = model.invoke(
            [{"role": "user", "content": prompt}],
            config={"callbacks": [langfuse_handler]},
        )
        return {"messages": [HumanMessage(content=response.content)]}

    # Node: Generate answer
    def generate_answer(state: MessagesState):
        """Generate an answer."""
        question = cast(str, state["messages"][0].content)
        context = cast(str, state["messages"][-1].content)
        prompt = rag_generate_prompt.compile(
            GenerateVariables(question=question, context=context)
        )
        response = model.invoke(
            [{"role": "user", "content": prompt}],
            config={"callbacks": [langfuse_handler]},
        )
        return {"messages": [response]}

    # Build graph
    workflow = StateGraph(MessagesState)

    # Define nodes
    workflow.add_node(generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node(rewrite_question)
    workflow.add_node(generate_answer)

    # Define edges
    workflow.add_edge(START, "generate_query_or_respond")

    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )

    # Grade documents after retrieval
    workflow.add_conditional_edges("retrieve", grade_documents)
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    return workflow.compile()


@pixie.app
async def langgraph_rag_agent(question: str) -> str:
    """Agentic RAG system that can answer questions about Lilian Weng's blog posts.

    The agent:
    1. Decides whether to retrieve or respond directly
    2. Grades retrieved documents for relevance
    3. Rewrites questions if needed
    4. Generates answers based on context

    Args:
        question: Natural language question about the blog content

    Returns:
        AI-generated answer based on retrieved context
    """
    # Setup retriever (this will take a moment on first run)
    retriever = setup_vectorstore()

    # Initialize model
    model = init_chat_model("gpt-4o-mini", temperature=0)

    # Create graph
    graph = create_rag_graph(retriever, model)

    print(f"Processing question: {question}")

    # Run the graph
    result = graph.invoke(
        {"messages": [{"role": "user", "content": question}]},  # type: ignore
        config={"callbacks": [langfuse_handler]},
    )

    # Return the final answer
    return result["messages"][-1].content
