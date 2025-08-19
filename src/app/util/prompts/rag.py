from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from loguru import logger

from app.util import llm

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
in its original language, which is English.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""  # noqa: E501

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


def _format_chat_history(chat_history: list[BaseMessage]) -> list:
    buffer = []
    logger.info(f" history :: {chat_history}")
    for message in chat_history:
        if isinstance(message, HumanMessage):
            buffer.append(message.content)
        elif isinstance(message, AIMessage):
            buffer.append(message.content)
        else:
            logger.warning(
                f"Unexpected message type in chat_history: {
                    type(message)}"
            )
    return buffer


_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up
    # question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(  # type: ignore
            run_name="HasChatHistoryCheck"
        ),
        # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(
                x["chat_history"]))
        | CONDENSE_QUESTION_PROMPT
        | llm.get_llm_instance()
        | StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(lambda x: x["question"]),
)

template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be detailed and concise.
Answer:"""

messages = ChatPromptTemplate.from_template(template)
