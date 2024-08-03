from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_remove_lucene_chars():
    with patch(
        "langchain_community.vectorstores.neo4j_vector.remove_lucene_chars"
    ) as mock:
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
        MagicMock(page_content="Mocked page content")
    ]
    return store_mock
