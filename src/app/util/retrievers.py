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


# Fulltext index query
def structured_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned in the question.
    """
    result = ""
    from data.store import get_default_store
    store = get_default_store()
    
    try:
        # Retrieve named entities from the question
        entities = get_ner_chain().invoke({"input": question})
        
        # Handle different response types
        entity_names = []
        
        if hasattr(entities, 'names'):
            # Expected structured response
            entity_names = entities.names
        elif hasattr(entities, 'tool_calls') and entities.tool_calls:
            # Tool calling response
            for tool_call in entities.tool_calls:
                if hasattr(tool_call, 'args'):
                    if hasattr(tool_call.args, 'names'):
                        entity_names.extend(tool_call.args.names)
                    elif hasattr(tool_call.args, 'entities'):
                        entity_names.extend(tool_call.args.entities)
        elif hasattr(entities, 'content'):
            # AIMessage response - parse the content
            logger.info(f"Got AIMessage content: {entities.content}")
            content = entities.content.strip()
            
            # Try to extract entities from the text response
            # This is a simple parser - you might need to adjust based on actual output format
            if content:
                # Look for common patterns like JSON, lists, or comma-separated values
                import re
                import json
                
                # Try JSON first
                try:
                    if content.startswith('{') or content.startswith('['):
                        parsed = json.loads(content)
                        if isinstance(parsed, dict):
                            entity_names = parsed.get('names', parsed.get('entities', []))
                        elif isinstance(parsed, list):
                            entity_names = parsed
                except json.JSONDecodeError:
                    pass
                
                # Try comma-separated or newline-separated
                if not entity_names:
                    # Look for patterns like "Entities: name1, name2, name3"
                    patterns = [
                        r'(?:entities|names):\s*([^\n]+)',
                        r'(?:entities|names)\s*=\s*([^\n]+)',
                        r'\[([^\]]+)\]',  # Look for [entity1, entity2]
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, content.lower())
                        if match:
                            entities_str = match.group(1)
                            entity_names = [e.strip().strip('"\'') for e in entities_str.split(',')]
                            break
                
                # Fallback: extract capitalized words
                if not entity_names:
                    entity_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        
        # Final fallback: extract from question itself
        if not entity_names:
            logger.warning("No entities extracted, using question words as fallback")
            words = question.split()
            entity_names = [word.strip('.,!?') for word in words if word.strip('.,!?').istitle() and len(word) > 2]
        
        logger.info(f"Extracted entities: {entity_names}")
        
        # Process each entity
        for entity in entity_names:
            if not entity or not isinstance(entity, str):
                continue
                
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
                MATCH (node)<-[r:!MENTIONS]-(neighbor)
                RETURN
                    '(id: ' + neighbor.id + ', description: ' + coalesce(neighbor.description, 'N/A') + ') ' +
                    '-[' + type(r) + ' (description: ' + coalesce(r.description, 'N/A') + ') ]-> ' +
                    '(id: ' + node.id + ', description: ' + coalesce(node.description, 'N/A') + ')' AS output
            }
            RETURN output LIMIT 50
            """
            
            try:
                # Execute the query and get the response
                response = store.graph.query(
                    query, {"query": generate_full_text_query(entity)}
                )
                
                # Append the query results to the final result
                result += "\n".join([el["output"] for el in response])
                
            except Exception as e:
                logger.error(f"Error querying entity '{entity}': {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error in structured_retriever: {e}")
        return ""
    
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
    {"#Document ".join(unstructured_data)}
   """
    return final_data
