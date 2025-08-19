from langchain_neo4j import Neo4jVector
from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars
from loguru import logger

from app.core.config import LLMSettings
from app.util.chains import get_ner_chain
from data.store import get_default_store


def get_graph_instance():
    store = get_default_store()
    return store.graph


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


def structured_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned in the question.

    Args:
        question: The input question to extract entities from

    Returns:
        Formatted string containing entity neighborhoods, or empty string on error
    """
    if not question or not question.strip():
        logger.warning("Empty question provided to structured_retriever")
        return ""

    from data.store import get_default_store

    store = get_default_store()

    try:
        entity_names = _extract_entities_from_question(question)
        logger.info(f"Extracted entities: {entity_names}")

        if not entity_names:
            logger.warning("No entities found in question")
            return ""

        return _query_entity_neighborhoods(store, entity_names)

    except Exception as e:
        logger.error(f"Error in structured_retriever: {e}", exc_info=True)
        return ""


def _extract_entities_from_question(question: str) -> List[str]:
    """Extract named entities from the question using various strategies."""
    try:
        entities = get_ner_chain().invoke({"input": question})
        entity_names = _parse_entity_response(entities)

        # Fallback: extract from question itself if no entities found
        if not entity_names:
            logger.warning(
                "No entities extracted from NER, using fallback extraction")
            entity_names = _fallback_entity_extraction(question)

        # Clean and deduplicate entities
        return _clean_entity_names(entity_names)

    except Exception as e:
        logger.error(f"Error extracting entities: {e}")
        return _fallback_entity_extraction(question)


def _parse_entity_response(entities) -> List[str]:
    """Parse entities from different response types."""
    entity_names = []

    if hasattr(entities, "names"):
        entity_names = entities.names
    elif hasattr(entities, "tool_calls") and entities.tool_calls:
        entity_names = _extract_from_tool_calls(entities.tool_calls)
    elif hasattr(entities, "content"):
        entity_names = _parse_content_response(entities.content)

    return entity_names


def _extract_from_tool_calls(tool_calls) -> List[str]:
    """Extract entity names from tool calling responses."""
    entity_names = []
    for tool_call in tool_calls:
        if hasattr(tool_call, "args"):
            if hasattr(tool_call.args, "names"):
                entity_names.extend(tool_call.args.names)
            elif hasattr(tool_call.args, "entities"):
                entity_names.extend(tool_call.args.entities)
    return entity_names


def _parse_content_response(content: str) -> List[str]:
    """Parse entities from AIMessage content using various strategies."""
    if not content:
        return []

    content = content.strip()
    entity_names = []

    # Try JSON parsing first
    entity_names = _try_json_parsing(content)
    if entity_names:
        return entity_names

    # Try regex patterns
    entity_names = _try_regex_patterns(content)
    if entity_names:
        return entity_names

    # Fallback: extract capitalized words
    return re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", content)


def _try_json_parsing(content: str) -> List[str]:
    """Attempt to parse entities from JSON content."""
    try:
        if content.startswith(("{", "[")):
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed.get("names", parsed.get("entities", []))
            elif isinstance(parsed, list):
                return parsed
    except json.JSONDecodeError:
        pass
    return []


def _try_regex_patterns(content: str) -> List[str]:
    """Try to extract entities using regex patterns."""
    patterns = [
        r"(?:entities|names):\s*([^\n]+)",
        r"(?:entities|names)\s*=\s*([^\n]+)",
        r"\[([^\]]+)\]",  # Look for [entity1, entity2]
    ]

    for pattern in patterns:
        match = re.search(pattern, content.lower())
        if match:
            entities_str = match.group(1)
            return [e.strip().strip("\"'") for e in entities_str.split(",")]

    return []


def _fallback_entity_extraction(question: str) -> List[str]:
    """Fallback method to extract potential entities from question text."""
    words = question.split()
    return [word.strip(".,!?") for word in words if word.strip(
        ".,!?").istitle() and len(word) > 2]


def _clean_entity_names(entity_names: List[str]) -> List[str]:
    """Clean and deduplicate entity names."""
    cleaned = []
    seen = set()

    for entity in entity_names:
        if not entity or not isinstance(entity, str):
            continue

        # Clean the entity name
        cleaned_entity = entity.strip().strip("\"'")
        if cleaned_entity and cleaned_entity.lower() not in seen:
            cleaned.append(cleaned_entity)
            seen.add(cleaned_entity.lower())

    return cleaned


def _query_entity_neighborhoods(store, entity_names: List[str]) -> str:
    """Query the graph database for entity neighborhoods."""
    # Optimized query with better performance
    query = """
    CALL db.index.fulltext.queryNodes('entity', $query, {limit: 3})
    YIELD node, score
    CALL (node) {
        WITH node
        OPTIONAL MATCH (node)-[r_out:!MENTIONS]->(neighbor_out)
        OPTIONAL MATCH (node)<-[r_in:!MENTIONS]-(neighbor_in)

        WITH node, r_out, neighbor_out, r_in, neighbor_in
        WHERE neighbor_out IS NOT NULL OR neighbor_in IS NOT NULL

        RETURN
            CASE
                WHEN neighbor_out IS NOT NULL THEN
                    'Node: ' + node.id + ' (' + coalesce(node.description, 'N/A') + ') ' +
                    '-[' + type(r_out) + ']-> ' + neighbor_out.id + ' (' + coalesce(neighbor_out.description, 'N/A') + ')'
                WHEN neighbor_in IS NOT NULL THEN
                    neighbor_in.id + ' (' + coalesce(neighbor_in.description, 'N/A') + ') ' +
                    '-[' + type(r_in) + ']-> ' + node.id + ' (' + coalesce(node.description, 'N/A') + ')'
            END AS output
    }
    RETURN DISTINCT output
    ORDER BY output
    LIMIT 50
    """

    results = []

    for entity in entity_names:
        try:
            response = store.graph.query(
                query, {"query": generate_full_text_query(entity)})

            entity_results = [el["output"] for el in response if el["output"]]
            if entity_results:
                results.append(f"--- Entity: {entity} ---")
                results.extend(entity_results)
                results.append("")  # Empty line for readability
            else:
                logger.info(f"No results found for entity: {entity}")

        except Exception as e:
            logger.error(f"Error querying entity '{entity}': {e}")
            continue

    return "\n".join(results).strip()


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
    {"#Document ".join(unstructured_data)}
   """
    return final_data
