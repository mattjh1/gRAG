import hashlib
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, cast

from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.vectorstores.neo4j_vector import SearchType
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
from loguru import logger
from tika import parser
from tqdm import tqdm

from data.graph_transformer_settings import GraphTransformerSettings, default_settings
from data.processors import semantic_split

TIKA_SERVER_URL: str = os.environ["TIKA_SERVER_URL"]


def get_checksum(obj: object) -> str:
    """Return checksum of the specific object (will be encoded to bytes)."""
    b: bytes
    if isinstance(obj, bytes):
        b = obj
    elif isinstance(obj, str):
        b = obj.encode()
    else:
        b = json.dumps(obj).encode()
    return hashlib.md5(b).hexdigest()


def parse_file(path: Path) -> list[Document]:
    chunks = []
    with path.open("rb") as src:
        if doc := parse_file_content(src.read()):
            content_type = "text"
            doc.metadata["content_type"] = content_type
            chunks = semantic_split(doc)

    logger.info(f"chunk len :: {len(chunks)}")
    if len(chunks) >= 3:
        sample_chunks = random.sample(chunks, 3)
        formatted_chunks = "\n\n".join(
            f"Sample {
                i +
                1}:\n{
                chunk.page_content}"
            for i, chunk in enumerate(sample_chunks)
        )
        logger.info(f"Chunk excerpt:\n\n{formatted_chunks}")
    return chunks


def parse_file_content(content: bytes) -> Optional[Document]:
    if len(content) <= 0:
        return None
    result = parser.from_buffer(
        content,
        serverEndpoint=TIKA_SERVER_URL,
    )
    result = cast(dict, result)
    text = result["content"]
    if text is None:
        return None
    checksum = get_checksum(content)
    doc = Document(
        page_content=text,
        metadata={"content_checksum": checksum},
    )
    logger.info(f"Full doc len :: {len(doc.page_content)} chars")
    return doc


def process_document(
    document: Document, llm_transformer: LLMGraphTransformer
) -> list[GraphDocument]:
    return llm_transformer.convert_to_graph_documents([document])


def as_graph_documents(
    page: Path, settings: GraphTransformerSettings = default_settings
) -> list[GraphDocument]:
    """Convert files at paths to LangChain GraphDocument object."""
    from app.util import get_ollama_instance

    docs = parse_file(page)
    if not docs:
        return []

    # llm = get_llm_instance()
    llm = get_ollama_instance()
    llm_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=settings["allowed_nodes"],
        allowed_relationships=settings["allowed_relationships"],
        node_properties=settings["node_properties"],
        relationship_properties=settings["relationship_properties"],
    )

    max_workers = 10
    graph_documents = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_document,
                doc,
                llm_transformer) for doc in docs]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing documents for neo4j",
        ):
            try:
                graph_document = future.result()
                graph_documents.extend(graph_document)
            except Exception as e:
                logger.error(f"Error processing document: {e}")

    return graph_documents


def as_vectors_from_graph(emb_model: Embeddings):
    from data.store import get_default_store

    store = get_default_store().vectorstore
    store.from_existing_graph(
        emb_model,
        search_type=SearchType.HYBRID,
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding",
        index_name=os.getenv("NEO4J_VECTOR_INDEX", "vector"),
        keyword_index_name=os.getenv("NEO4J_KEYWORD_INDEX", "keyword"),
    )
