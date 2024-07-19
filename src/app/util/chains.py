from operator import itemgetter
from typing import Optional, Tuple

from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from loguru import logger

from app.core.config import LLMSettings, config


def get_memory() -> Tuple[ConversationBufferMemory, Runnable]:
    memory = ConversationBufferMemory(
        return_messages=False,
        input_key="human",
        output_key="ai",
    )

    memory_loader = RunnablePassthrough.assign(
        history=RunnableLambda(
            func=memory.load_memory_variables,
            name="Load Memory Variables",
        )
        | itemgetter(memory.memory_key)
    )

    return memory, memory_loader


def get_ner_chain(llm_settings: Optional[LLMSettings] = None) -> Runnable:
    from app.util import get_llm_instance, get_ollama_instance, prompts

    settings = llm_settings or config.LLM_SETTINGS
    logger.info(f"settings :: {settings}")

    if settings.provider == "ollama":
        llm = get_ollama_instance(settings)
    else:
        llm = get_llm_instance(settings)

    return prompts.entities.messages | llm.with_structured_output(
        prompts.entities.Entities
    )


def get_rag_chain(llm_settings: Optional[LLMSettings] = None) -> Runnable:
    from app.util import get_llm_instance, prompts, retrievers

    return (
        RunnableParallel(
            {
                "context": prompts.rag._search_query | retrievers.super_retriever,
                "question": RunnablePassthrough(),
            }
        )
        | prompts.rag.messages
        | get_llm_instance(llm_settings)
        | StrOutputParser()
    )


def get_summary_chain(llm_settings: Optional[LLMSettings] = None) -> Runnable:
    from app.util import get_llm_instance, prompts

    llm = get_llm_instance(llm_settings)
    return prompts.summary.messages | llm


def get_condense_chain(llm_settings: Optional[LLMSettings] = None) -> Runnable:
    from app.util import get_llm_instance, get_ollama_instance, prompts

    settings = llm_settings or config.LLM_SETTINGS

    if settings.provider == "ollama":
        llm = get_ollama_instance(settings)
    else:
        llm = get_llm_instance(settings)
    _, memory_loader = get_memory()

    return (
        memory_loader
        | prompts.entities.messages
        | llm.with_structured_output(prompts.entities.Entities)
    )
