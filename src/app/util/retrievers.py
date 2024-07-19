from langchain_community.vectorstores.neo4j_vector import (
    Neo4jVector,
    remove_lucene_chars,
)
from loguru import logger

from app.core.config import LLMSettings
from app.util.chains import get_ner_chain


def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspelings.
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()


# Fulltext index query
def structured_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned in the question.
    """
    result = ""
    from data.store import get_default_store

    store = get_default_store()

    # Retrieve named entities from the question
    entities = get_ner_chain(LLMSettings(provider="ollama", model="phi3:14b")).invoke(
        {"input": question}
    )

    for entity in entities.names:
        query = """
        CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
        YIELD node, score
        CALL {
            WITH node
            MATCH (node)-[r:!MENTIONS]->(neighbor)
            RETURN
                'Node: (id: ' + node.id + ', description: ' + coalesce(node.description, 'N/A') + ') ' +
                '-[' + type(r) + ' (description: ' + coalesce(r.description, 'N/A') + ') ]-> ' +
                '(id: ' + neighbor.id + ', description: ' + coalesce(neighbor.description, 'N/A') + ')' AS output
            UNION ALL
            WITH node
            MATCH (node)<-[r:MENTIONS]-(neighbor)
            RETURN
                '(id: ' + neighbor.id + ', description: ' + coalesce(neighbor.description, 'N/A') + ') ' +
                '-[' + type(r) + ' (description: ' + coalesce(r.description, 'N/A') + ') ]-> ' +
                '(id: ' + node.id + ', description: ' + coalesce(node.description, 'N/A') + ')' AS output
        }
        RETURN output LIMIT 50
        """

        # Execute the query and get the response
        response = store.graph.query(query, {"query": generate_full_text_query(entity)})

        # Append the query results to the final result
        result += "\n".join([el["output"] for el in response])

    return result


def hybrid_retriever() -> Neo4jVector:
    from data.store import get_default_store

    store = get_default_store()
    return store.vectorstore.from_existing_index(
        store.embeddings, index_name="vector", keyword_index_name="keyword"
    )


def super_retriever(question: str) -> str:
    logger.info(f"Search query: {question}")

    structured_data = structured_retriever(question)

    hybrid_result = hybrid_retriever().similarity_search(question, k=2)
    unstructured_data = [el.page_content for el in hybrid_result]

    final_data = f"""Structured data:
    {structured_data}
    Unstructured data:
    {"#Document ". join(unstructured_data)}
   """
    return final_data
