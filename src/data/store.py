import functools
from enum import Enum

from langchain_neo4j import Neo4jGraph
from langchain_neo4j.graphs.graph_document import GraphDocument
from langchain_neo4j import Neo4jVector
from langchain_core.embeddings import Embeddings
from loguru import logger

from app.core.config import config


class StoreEnum(str, Enum):
    neo4j = "neo4j"
    redis = "redis"


class Store:
    def __init__(
        self,
        graph: Neo4jGraph,
        vectorstore: Neo4jVector,
        embeddings: Embeddings,
    ):
        self.graph = graph
        self.vectorstore = vectorstore
        self.embeddings = embeddings

    def store_graph(self, docs: list[GraphDocument]) -> None:
        self.graph.add_graph_documents(docs, baseEntityLabel=True, include_source=True)
        self.graph.query(
            "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]"
        )
        self.graph.refresh_schema()
        logger.info(f"Graph schema :: {self.graph.schema}")


@functools.lru_cache(maxsize=1)
def get_default_store() -> Store:
    from langchain_ollama import OllamaEmbeddings

    graph = Neo4jGraph(
        url=config.NEO4J_URI,
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
    )

    embeddings = OllamaEmbeddings(
        base_url=config.OLLAMA_API_BASE,
        model=config.EMB_MODEL_ID,
    )

    vectorstore = Neo4jVector(
        url=config.NEO4J_URI,
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        embedding=embeddings,
    )

    return Store(
        graph=graph,
        embeddings=embeddings,
        vectorstore=vectorstore,
    )
