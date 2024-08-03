from typing import Any

import chainlit as cl
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import Runnable, RunnableConfig
from loguru import logger

from app.util import get_memory, get_rag_chain, get_summary_chain


def set_session(key: str, value: Any) -> None:
    """Set a value in the user session."""
    return cl.user_session.set(key, value)


def get_session(key: str) -> Any:
    """Get a value from the user session."""
    default_ = object()
    value = cl.user_session.get(key, default=default_)
    if value == default_:
        raise Exception(f"Object '{key}' not found in `cl.user_session`.")
    return value


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content="""
Hi I'm a bot specifically trained on the novel "Dracula" by Bram Stoker.

Please ask me anything related to this book.

## Example questions

- **Describe the role of Renfield in the novel.**
- **Analyze the theme of modernity versus antiquity in "Dracula".**
- **Compare and contrast the characters of Jonathan Harker and Dr. John Seward in terms of their roles and character development.**
"""
    ).send()

    memory, _ = get_memory()
    rag_chain = get_rag_chain()
    summary_chain = get_summary_chain()

    set_session("memory", memory)
    set_session("rag_chain", rag_chain)
    set_session("summary_chain", summary_chain)


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages."""
    memory: ConversationBufferMemory = get_session("memory")
    rag_chain: Runnable = get_session("rag_chain")
    query = {"question": message.content}

    if memory.chat_memory.messages:
        query = {
            "question": message.content,
            "chat_history": memory.chat_memory.messages,
        }

    msg = cl.Message(content="")
    answer: str = ""

    async for chunk in rag_chain.astream(
        query,
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        answer += chunk
        await msg.stream_token(chunk)

        # if "answer" in chunk:
        #     answer += chunk["answer"]
        #     await msg.stream_token(chunk["answer"])

    await msg.update()
    logger.info(f"answer :: {answer}")

    memory.save_context(inputs={"human": message.content}, outputs={"ai": answer})
