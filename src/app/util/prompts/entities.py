# Extract entities from text
from langchain_core.prompts.chat import ChatPromptTemplate
from pydantic import BaseModel, Field


class Entities(BaseModel):
    """Identifying information about entities."""

    names: list[str] = Field(
        ...,
        description="All the pronoun entities that appear in the text",
    )


messages = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are extracting pronoun entities from the text, such as people, places, and organizations. If the entities are not in english, translate to english.",
         ),
        ("human",
         "Use the given format to extract information from the following "
         "input: {input}",
         ),
    ])
