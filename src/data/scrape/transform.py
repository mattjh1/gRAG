import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Generator, Union

from langchain_community.graphs.graph_document import GraphDocument
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from loguru import logger
from tqdm import tqdm

from app.util import get_llm_instance
from data.graph_transformer_settings import GraphTransformerSettings, default_settings
from data.processors import split
from data.store import StoreEnum


class Transformer:
    """
    A class to handle the transformation of content into Document instances with metadata.
    """

    def __init__(self, database: StoreEnum) -> None:
        self.database = database
        self.llm = get_llm_instance()

    @staticmethod
    def get_checksum(obj: object) -> str:
        """
        Return checksum of the specific object (will be encoded to bytes).

        Args:
            obj (object): The object to calculate the checksum for.

        Returns:
            str: The MD5 checksum of the object.
        """
        b: bytes
        if isinstance(obj, bytes):
            b = obj
        elif isinstance(obj, str):
            b = obj.encode()
        else:
            b = json.dumps(obj).encode()
        return hashlib.md5(b).hexdigest()

    def process_document(
        self, document: Document, llm_transformer: LLMGraphTransformer
    ) -> list[GraphDocument]:
        return llm_transformer.convert_to_graph_documents([document])

    def as_documents(
        self,
        data: Generator[dict[str, Any], None, None],
        settings: GraphTransformerSettings = default_settings,
    ) -> Union[list[GraphDocument], list[Document]]:
        if self.database == StoreEnum.redis:
            return self._as_documents(data)
        elif self.database == StoreEnum.neo4j:
            return self._as_graph_documents(data, settings)
        else:
            raise ValueError("Database not supported")

    def _as_documents(
            self, data: Generator[dict[str, Any], None, None]) -> list[Document]:
        return list(self.transform_all(data))

    def _as_graph_documents(
        self,
        data: Generator[dict[str, Any], None, None],
        settings: GraphTransformerSettings = default_settings,
    ) -> list[GraphDocument]:
        docs = self.transform_all(data)

        llm_transformer = LLMGraphTransformer(
            llm=self.llm,
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
                    self.process_document,
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

    def transform(self, content: str,
                  metadata: dict) -> Union[list[Document], Document]:
        """
        Transform content and metadata into Document instances.

        Args:
            content (str): The content to be transformed.
            metadata (dict): Metadata associated with the content.

        Returns:
            Union[list[Document], Document]: A single Document or a list of Documents.
        """
        docs = []
        if metadata["length"] > 512:
            _doc = Document(
                page_content=content,
            )
            split_contents = split(_doc)
            for part in split_contents:
                checksum = self.get_checksum(part.page_content)
                doc = Document(
                    page_content=part.page_content,
                    metadata={"content_checksum": checksum, **metadata},
                )
                docs.append(doc)
        else:
            checksum = self.get_checksum(content)
            doc = Document(
                page_content=content,
                metadata={"content_checksum": checksum, **metadata},
            )
            return doc
        return docs

    def transform_all(
        self, data: Generator[dict[str, Any], None, None]
    ) -> Generator[Document, None, None]:
        """
        Transform all content and metadata in the generator into Document instances.

        Args:
            data (Generator[dict[str, Any], None, None]): A generator yielding dictionaries with content and metadata.

        Yields:
            Generator[Document, None, None]: A generator yielding Document instances.
        """
        for item in data:
            content = item["content"]
            metadata = item["metadata"]
            documents = self.transform(content, metadata)
            if isinstance(documents, Document):
                yield documents
            else:
                for document in documents:
                    yield document
