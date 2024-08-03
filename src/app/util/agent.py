from typing import List, Tuple

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_messages
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from loguru import logger

from app.util import get_llm_instance, tools
from app.util.prompts import react

llm = get_llm_instance()
llm_with_stop = llm.bind(stop=["\nObservation"])


def _format_chat_history(chat_history: list[BaseMessage]) -> list:
    buffer = []
    logger.info(f" history :: {chat_history}")
    for message in chat_history:
        if isinstance(message, HumanMessage):
            buffer.append(message.content)
        elif isinstance(message, AIMessage):
            buffer.append(message.content)
        else:
            logger.warning(f"Unexpected message type in chat_history: {type(message)}")
    return buffer


agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_messages(x["intermediate_steps"]),
        "chat_history": lambda x: (
            _format_chat_history(x["chat_history"]) if x.get("chat_history") else []
        ),
    }
    | react.messages
    | llm_with_stop
    | ReActJsonSingleInputOutputParser()
)


# Add typing for input
class AgentInput(BaseModel):
    input: str
    chat_history: List[Tuple[str, str]] = Field(
        ..., extra={"widget": {"type": "chat", "input": "input", "output": "output"}}
    )


agent_executor = AgentExecutor(agent=agent, tools=tools).with_types(
    input_type=AgentInput
)  # type: ignore
