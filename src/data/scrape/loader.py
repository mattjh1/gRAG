import os
from typing import Union

from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.vectorstores import Redis
from langchain_community.vectorstores.neo4j_vector import SearchType
from langchain_core.documents import Document

from data.store import get_default_store


class DataLoader:
    """
    A class to handle loading transformed data into the target storage.
    """

    def __init__(self) -> None:
        self.db = get_default_store()

    def load(
        self, transformed_data: Union[list[GraphDocument], list[Document]]
    ) -> None:
        """
        Loads the transformed data into the target storage.

        Args:
            transformed_data (Union[List[GraphDocument], List[Document]]): The data to load.
        """
        if all(isinstance(doc, GraphDocument) for doc in transformed_data):
            self._load_graph_documents(transformed_data)  # type: ignore
        elif all(isinstance(doc, Document) for doc in transformed_data):
            self._load_documents(transformed_data)  # type: ignore
        else:
            raise ValueError(
                "transformed_data must be a list of GraphDocument or Document instances."
            )

    def _load_documents(self, docs: list[Document]) -> None:
        _ = Redis.from_documents(
            documents=docs,
            embedding=self.db.embeddings,
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            index_name="redis-index",
            schema="../schema.yaml",
        )

    def _load_graph_documents(self):
        self.db.vectorstore.from_existing_graph(
            self.db.embeddings,
            search_type=SearchType.HYBRID,
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding",
            index_name=os.getenv("NEO4J_VECTOR_INDEX", "vector"),
            keyword_index_name=os.getenv("NEO4J_KEYWORD_INDEX", "keyword"),
        )
