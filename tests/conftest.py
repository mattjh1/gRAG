from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_remove_lucene_chars():
    with patch("langchain_community.vectorstores.neo4j_vector.remove_lucene_chars") as mock:
        yield mock


@pytest.fixture
def mock_get_default_store():
    with patch("data.store.get_default_store") as mock:
        yield mock


@pytest.fixture
def mock_get_ner_chain():
    with patch("app.util.chains.get_ner_chain") as mock:
        yield mock


@pytest.fixture
def mock_store():
    store_mock = MagicMock()
    store_mock.graph.query.return_value = [{"output": "Mocked output"}]
    store_mock.vectorstore.from_existing_index.return_value.similarity_search.return_value = [
        MagicMock(page_content="Mocked page content")]
    return store_mock


@pytest.fixture
def mock_env_vars(monkeypatch):
    env_vars = {
        "MODE": "development",
        "LLM_SETTINGS_PROVIDER": "ollama",
        "LLM_SETTINGS_MODEL": "phi3:14b",
        "LLM_SETTINGS_TEMPERATURE": "0.0",
        "PROJECT_NAME": "rag-api",
        "API_STR": "/api",
        "OPENROUTER_API_BASE": "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY": "mocked_openrouter_api_key",
        "OLLAMA_API_BASE": "http://localhost:11434",
        "EMB_MODEL_ID": "nomic-embed-text",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "mocked_neo4j_username",
        "NEO4J_PASSWORD": "mocked_neo4j_password",
        "PROVIDERS_OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
        "PROVIDERS_OPENROUTER_API_KEY": "mocked_providers_openrouter_api_key",
        "PROVIDERS_OLLAMA_BASE_URL": "http://localhost:11434/v1",
        "PROVIDERS_OLLAMA_API_KEY": "mocked_providers_ollama_api_key",
        "PROVIDERS_OPENAI_BASE_URL": "https://api.openai.com",
        "PROVIDERS_OPENAI_API_KEY": "mocked_providers_openai_api_key",
        "BACKEND_CORS_ORIGINS": "*",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
