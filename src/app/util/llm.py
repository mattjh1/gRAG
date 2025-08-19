from typing import Optional

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app.core.config import LLMSettings, config


class LLMClientManager:
    _instance: Optional[ChatOpenAI] = None
    _ollama_instance: Optional[ChatOllama] = None

    @classmethod
    def get_instance(
            cls,
            settings: Optional[LLMSettings] = None) -> ChatOpenAI:
        if cls._instance is None:
            cls._instance = cls._create_llm_client(settings)
        return cls._instance

    @classmethod
    def get_ollama_instance(
            cls,
            settings: Optional[LLMSettings] = None) -> ChatOllama:
        if cls._ollama_instance is None:
            cls._ollama_instance = cls._create_ollama_client(settings)
        return cls._ollama_instance

    @classmethod
    def _create_llm_client(
            cls,
            settings: Optional[LLMSettings] = None) -> ChatOpenAI:
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
            cls,
            settings: Optional[LLMSettings] = None) -> ChatOllama:
        settings = settings or config.LLM_SETTINGS

        if settings.provider == "ollama":
            options = {
                "base_url": config.OLLAMA_API_BASE,
                "model": settings.model,
                "temperature": settings.temperature,
            }
            return ChatOllama(**options)
        else:
            raise ValueError(
                "Provider must be set to `ollama` to get `ollama` client")


get_llm_instance = LLMClientManager.get_instance

get_ollama_instance = LLMClientManager.get_ollama_instance
