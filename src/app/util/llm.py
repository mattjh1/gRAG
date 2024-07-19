from typing import Optional

from langchain_openai import ChatOpenAI
from loguru import logger

from app.core.config import LLMSettings, config
from app.util.ollama_functions import OllamaFunctions


class LLMClientManager:
    _instance: Optional[ChatOpenAI] = None
    _ollama_instance: Optional[OllamaFunctions] = None

    @classmethod
    def get_instance(cls, settings: Optional[LLMSettings] = None) -> ChatOpenAI:
        if cls._instance is None:
            cls._instance = cls._create_llm_client(settings)
        return cls._instance

    @classmethod
    def get_ollama_instance(
        cls, settings: Optional[LLMSettings] = None
    ) -> OllamaFunctions:
        if cls._ollama_instance is None:
            cls._ollama_instance = cls._create_ollama_client(settings)
        return cls._ollama_instance

    @classmethod
    def _create_llm_client(cls, settings: Optional[LLMSettings] = None) -> ChatOpenAI:
        settings = settings or config.LLM_SETTINGS

        provider = config.PROVIDERS[settings.provider]
        options = {
            "base_url": provider["base_url"],
            "api_key": provider["api_key"],
            "model": settings.model,
            "temperature": settings.temperature,
        }

        return ChatOpenAI(**options)

    @classmethod
    def _create_ollama_client(
        cls, settings: Optional[LLMSettings] = None
    ) -> OllamaFunctions:
        settings = settings or config.LLM_SETTINGS

        if settings.provider == "ollama":
            options = {
                "base_url": config.OLLAMA_API_BASE,
                "model": settings.model,
                "temperature": settings.temperature,
            }
            return OllamaFunctions(format="json", **options)
        else:
            raise ValueError("Provider must be set to `ollama` to get `ollama` client")


get_llm_instance = LLMClientManager.get_instance

# use this hack to use openai function calling with ollama models
get_ollama_instance = LLMClientManager.get_ollama_instance
