import uuid
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger

from app.core.config import LLMSettings, config


# NEW: Modern conversation state replacing ConversationBufferMemory
class ConversationState:
    """Modern replacement for ConversationBufferMemory using LangGraph."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self._messages: List[BaseMessage] = []
        self.memory_saver = MemorySaver()
        self.memory_key = "history"  # Default key for compatibility
        self.input_key = "human"
        self.output_key = "ai"
        self.return_messages = False

    def add_message(self, message: BaseMessage):
        """Add a message to conversation history."""
        self._messages.append(message)

    def get_messages(self) -> List[BaseMessage]:
        """Get all messages in conversation."""
        return self._messages.copy()

    def save_context(self, inputs: Dict[str, str], outputs: Dict[str, str]):
        """Save context (maintains API compatibility with ConversationBufferMemory)."""
        if self.input_key in inputs:
            self.add_message(HumanMessage(content=inputs[self.input_key]))
        if self.output_key in outputs:
            self.add_message(AIMessage(content=outputs[self.output_key]))

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables (maintains API compatibility)."""
        if self.return_messages:
            return {self.memory_key: self._messages}
        else:
            # Convert messages to string format like the old
            # ConversationBufferMemory
            history_str = ""
            for message in self._messages:
                if isinstance(message, HumanMessage):
                    history_str += f"Human: {message.content}\n"
                elif isinstance(message, AIMessage):
                    history_str += f"AI: {message.content}\n"
            return {self.memory_key: history_str.strip()}

    def clear(self):
        """Clear conversation history."""
        self._messages = []

    @property
    def chat_memory(self):
        """Compatibility property for accessing messages."""
        return self

    @property
    def messages(self) -> List[BaseMessage]:
        """Compatibility property."""
        return self._messages


def get_memory() -> Tuple[ConversationState, Runnable]:
    """
    Updated memory function using ConversationState instead of ConversationBufferMemory.

    Returns:
        Tuple of (conversation_state, memory_loader) - same interface as before
    """
    # Create modern conversation state with same configuration as old memory
    conversation_state = ConversationState()
    conversation_state.return_messages = False
    conversation_state.input_key = "human"
    conversation_state.output_key = "ai"

    # Create memory loader with same interface as before
    memory_loader = RunnablePassthrough.assign(
        history=RunnableLambda(
            func=conversation_state.load_memory_variables,
            name="Load Memory Variables",
        )
        | itemgetter(conversation_state.memory_key)
    )

    return conversation_state, memory_loader


def get_ner_chain(llm_settings: Optional[LLMSettings] = None) -> Runnable:
    from app.util import get_llm_instance, get_ollama_instance, prompts

    settings = llm_settings or config.LLM_SETTINGS
    logger.info(f"settings :: {settings}")

    if settings.provider == "ollama":
        llm = get_ollama_instance(settings)
    else:
        llm = get_llm_instance(settings)

    return prompts.entities.messages | llm.bind_tools(
        tools=[prompts.entities.Entities])


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

    # Use the updated memory function
    _, memory_loader = get_memory()

    return (
        memory_loader
        | prompts.entities.messages
        | llm.bind_tools(tools=[prompts.entities.Entities])
    )
