from typing import TypedDict

from langchain.callbacks.manager import Callbacks
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import BaseTool

from app.util import chains
from app.util.retrievers import hybrid_retriever
from data.store import get_default_store


class DuckDuckGoResult(TypedDict):
    title: str
    snippet: str
    link: str


@tool("Graph Query Tool")
def graph_query_tool(query: str) -> str:
    """
    Useful for querying the Neo4j graph using Cypher.
    Args:
        query (str): The Cypher query to execute.
    Returns:
        str: The result of the query.
    """
    store = get_default_store()
    return store.graph.query(query)


@tool("Vector Search Tool")
def vector_search_tool(query: str) -> str:
    """
    Useful for performing a similarity search on the vector index.
    Args:
        query (str): The query to search for.
    Returns:
        str: A list of documents.
    """
    retriever = hybrid_retriever()
    result = retriever.similarity_search(query)
    return str(result)


@tool("Hybrid Search Tool")
def hybrid_search_tool(query: str) -> str:
    """
    Useful for performing a hybrid search on the vector index.
    Args:
        query (str): The query to search for.
    Returns:
        str: A list of documents.
    """
    retriever = hybrid_retriever()
    result = retriever.hybrid_search(query)
    return str(result)


@tool("Smalltalk", return_direct=True)
async def smalltalk(action_input: str) -> str:
    """Useful when you need to greet human, provide short answer, or offer your services.
    Considering using 'Resolve Ambiguity' or 'Check Memory' tool before using this tool
    to avoid asking unnecessary questions.
    """
    return action_input


@tool("Resolve Ambiguity")
async def resolve_ambiguity(
    action_input: str, callbacks: Callbacks | None = None
) -> str:
    """Useful when you need to resolve ambiguity (pronouns) in the question or text.
    The action input should be a question or text that contains ambiguous pronouns."""
    return await chains.get_condense_chain().ainvoke(
        {"input": action_input},
        {"callbacks": callbacks},
    )


@tool("Search Internet")
async def search_internet(action_input: str) -> str:
    """Useful when you need to search the internet for answers. Especially useful for
    current events or filling in missing information not found by other tools,
    consider this tool a last resort."""

    search = DuckDuckGoSearchRun()
    res = search.run(action_input)

    return res


@tool("Final answer", return_direct=True)
def finish(answer: str) -> str:
    """
    Returns the answer and finishes the task.
    Use this tool in case you cannot find the answer using the search tool.
    """
    return answer


tools: list[BaseTool] = [
    graph_query_tool,
    vector_search_tool,
    hybrid_search_tool,
    smalltalk,
    resolve_ambiguity,
    search_internet,
    finish,
]  # type: ignore


if __name__ == "__main__":
    import asyncio

    from loguru import logger

    question = "Who is Dracula?"
    res = asyncio.run(search_database(question))  # type: ignore
    logger.info(res)

