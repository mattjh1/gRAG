from langchain.docstore.document import Document
from langchain_community.graphs.graph_document import GraphDocument
from langchain_experimental.graph_transformers import LLMGraphTransformer


def as_graph_documents(pages: list[Document]) -> list[GraphDocument]:
    from app.util import get_llm_instance

    llm = get_llm_instance()
    llm_transformer = LLMGraphTransformer(llm=llm)
    graph_docs = llm_transformer.convert_to_graph_documents(pages)
    return graph_docs
