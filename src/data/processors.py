from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

"""Recursive splitter of text to max chunk size with overlap."""
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
)


def split(
    doc: Document,
) -> list[Document]:
    """Split document."""
    chunks = text_splitter.split_documents([doc])
    return chunks


def semantic_split(doc: Document) -> list[Document]:
    from data.store import get_default_store

    store = get_default_store()
    """semantic splitter of text using cosine similarity of sentences to set chunk size 
    """
    semantic_text_splitter = SemanticChunker(
        store.embeddings, breakpoint_threshold_type="percentile"
    )
    """split using semantic similarity"""
    chunks = semantic_text_splitter.split_documents([doc])
    return chunks
