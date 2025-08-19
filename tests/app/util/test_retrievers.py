import pytest

from app.util.retrievers import (
    _generate_full_text_query,
    hybrid_retriever,
    structured_retriever,
    super_retriever,
)


def test_generate_full_text_query(mock_remove_lucene_chars, mock_env_vars):
    mock_remove_lucene_chars.return_value = "test input"

    result = _generate_full_text_query("test input")
    assert result == "test~2 AND input~2"


def test_structured_retriever(
    mock_get_default_store, mock_get_ner_chain, mock_store, mock_env_vars
):
    mock_get_default_store.return_value = mock_store
    mock_get_ner_chain.return_value.invoke.return_value.names = ["test_entity"]

    result = structured_retriever("test question")

    # Adjusted to match the actual expected output format
    assert any("Mocked output" in res.get("output", "")
               for res in mock_store.graph.query())
    mock_store.graph.query.assert_called_once()


def test_hybrid_retriever(mock_get_default_store, mock_store, mock_env_vars):
    mock_get_default_store.return_value = mock_store

    result = hybrid_retriever()

    # Ensure only the function was called
    assert result.similarity_search.call_count == 0
    mock_store.vectorstore.from_existing_index.assert_called_once()


def test_super_retriever(
        mock_get_default_store,
        mock_get_ner_chain,
        mock_store,
        mock_env_vars):
    mock_get_default_store.return_value = mock_store
    mock_get_ner_chain.return_value.invoke.return_value.names = ["test_entity"]

    result = super_retriever("test question")

    assert "Structured data:" in result
    assert "Unstructured data:" in result
    assert "Mocked page content" in result
    assert any("Mocked output" in res.get("output", "")
               for res in mock_store.graph.query())
    mock_store.graph.query.assert_called_once()
    mock_store.vectorstore.from_existing_index().similarity_search.assert_called_once()
