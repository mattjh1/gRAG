from typing import TypedDict

from langchain.callbacks.manager import Callbacks
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from app.util import chains


class DuckDuckGoResult(TypedDict):
    title: str
    snippet: str
    link: str


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


class SearchDatabaseInput(BaseModel):
    entity: str = Field(description="Part or Sourcing Request mentioned in question")
    entity_type: str = Field(
        description="Type of entity. Possible options are Part or Sourcing Request"
    )


@tool("Search Database", args_schema=SearchDatabaseInput)
async def search_database(action_input: str, callbacks: Callbacks | None = None) -> str:
    """Useful when you want to search the knowledge database for any unknown topic.
    This tool will search the knowledge database and return complete and relevant answer.
    You can use this tools multiple times using different queries to refine the answer.
    The output from this tool can be used as Final Answer to the user's question.
    """
    # docs = await retrievers.neo4j_retriever.ainvoke(
    #     input=action_input, callbacks=callbacks
    # )
    # answer = "\n\n".join(d.page_content for d in docs)
    # return answer
    return ""


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
    smalltalk,
    resolve_ambiguity,
    search_database,
    search_internet,
    finish,
]  # type: ignore
