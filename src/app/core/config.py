import json
import os
from enum import Enum
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic import AnyHttpUrl, BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModeEnum(str, Enum):
    development = "development"
    production = "production"
    testing = "testing"


class LLMSettings(BaseModel):
    provider: Optional[str] = Field(
        default="openrouter", description="The provider for the LLM"
    )
    model: Optional[str] = Field(default=None, description="The model for the LLM")
    temperature: Optional[float] = Field(
        default=0, description="The temperature for the LLM"
    )

    @field_validator("model", mode="before")
    def validate_model(cls, v, values):
        provider = values.data.get("provider", "openrouter")
        openrouter_models = [
            "databricks/dbrx-instruct:nitro",
            "mistralai/mistral-7b-instruct",
            "mistralai/mixtral-8x7b-instruct",
            "openai/gpt-4o",
            "openai/gpt-4o-mini"
        ]
        openai_models = [
            "gpt-3.5-turbo",
            "gpt-4o",
            "gpt-4-turbo",
        ]
        ollama_models = [
            "mistral:7b-instruct",
            "gemma2:27b",
            "gemma2:9b",
            "wizardlm2:7b",
            "phi3:14b",
        ]

        # If model is None, set a default based on the provider
        if v is None:
            if provider == "openrouter":
                return "openai/gpt-4o"
            elif provider == "openai":
                return "gpt-4o"
            elif provider == "ollama":
                return "phi3:14b"

        if provider == "openrouter" and v not in openrouter_models:
            raise ValueError(
                f"For provider 'openrouter', model must be one of {openrouter_models}"
            )
        elif provider == "openai" and v not in openai_models:
            raise ValueError(
                f"For provider 'openai', model must be one of {openai_models}"
            )
        elif provider == "ollama" and v not in ollama_models:
            raise ValueError(
                f"For provider 'ollama', model must be one of {ollama_models}"
            )

        return v


class Config(BaseSettings):
    MODE: ModeEnum = ModeEnum.development
    LLM_SETTINGS: LLMSettings = LLMSettings(
        provider=os.getenv("LLM_API_PROVIDER", "openrouter"),
        model=os.getenv("LLM_MODEL_ID", "openai/gpt-4o"),
        temperature=0,
    )
    PROJECT_NAME: str = "rag-api"
    API_STR: str = "/api"
    OPENROUTER_API_BASE: str = os.environ["OPENROUTER_API_BASE"]
    OPENROUTER_API_KEY: str = os.environ["OPENROUTER_API_KEY"]
    OLLAMA_API_BASE: str = os.environ["OLLAMA_API_BASE"]
    EMB_MODEL_ID: str = os.environ["EMB_MODEL_ID"]
    NEO4J_URI: str = os.environ["NEO4J_URI"]
    NEO4J_USERNAME: str = os.environ["NEO4J_USERNAME"]
    NEO4J_PASSWORD: str = os.environ["NEO4J_PASSWORD"]
    PROVIDERS: dict = {
        "openrouter": {
            "base_url": OPENROUTER_API_BASE,
            "api_key": OPENROUTER_API_KEY,
        },
        # TODO: make OpenAI compatible
        "openai": {
            "base_url": "https://api.openai.com",
            "api_key": "your_openai_api_key",
        },
        # TODO: add ollama...
        "ollama": {"base_url": f"{OLLAMA_API_BASE}/v1", "api_key": "ollama"},
    }

    try:
        BACKEND_CORS_ORIGINS: list[str] | list[AnyHttpUrl] = json.loads(
            os.getenv("BACKEND_CORS_ORIGINS", "[]")
        )
    except json.JSONDecodeError as e:
        logger.error(f"Allowed CORS origin list is not formatted correctly: {e}")

    @field_validator("BACKEND_CORS_ORIGINS")
    def assemble_cors_origins(cls, v: str | list[str]) -> list[str] | str:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v

    model_config = SettingsConfigDict(case_sensitive=True, env_file=Path())

    def print_config(self) -> None:
        """
        Print the configuration settings to stdout, masking sensitive information.
        """
        config_dict = self.model_dump()
        if "OPENROUTER_API_KEY" in config_dict:
            config_dict["OPENROUTER_API_KEY"] = "***MASKED***"
        if "PROVIDERS" in config_dict:
            for provider in config_dict["PROVIDERS"].values():
                if "api_key" in provider:
                    provider["api_key"] = "***MASKED***"

        logger.info(json.dumps(config_dict, indent=2))


config = Config()
config.print_config()
