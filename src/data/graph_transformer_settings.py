from typing import TypedDict, Union


class GraphTransformerSettings(TypedDict):
    allowed_nodes: list[str]
    allowed_relationships: list[str]
    node_properties: Union[bool, list[str]]
    relationship_properties: Union[bool, list[str]]


default_settings: GraphTransformerSettings = {
    "allowed_nodes": [],
    "allowed_relationships": [],
    "node_properties": False,
    "relationship_properties": False,
}

ms_graphrag_settings: GraphTransformerSettings = {
    "allowed_nodes": [],
    "allowed_relationships": [],
    "node_properties": ["description"],
    "relationship_properties": ["description"],
}

dracula_settings: GraphTransformerSettings = {
    "allowed_nodes": [
        "Character",
        "Location",
        "Event",
        "Item",
        "Chapter",
        "DiaryEntry",
        "Concept",
    ],
    "allowed_relationships": [
        "KNOWS",
        "LOCATED_IN",
        "PARTICIPATES_IN",
        "HAPPENS_IN",
        "MENTIONS",
        "NEXT",
        "OWNS",
        "TRAVELS_TO",
        "THEME",
        "WRITTEN_BY",
    ],
    "node_properties": [
        "name",
        "role",
        "gender",
        "type",
        "description",
        "number",
        "title",
        "author",
        "date",
        "content",
    ],
    "relationship_properties": [
        "since",
        "condition",
        "role",
        "date",
        "context",
        "order",
        "acquired_date",
        "method",
        "importance",
        "date",
    ],
}

# more graph settings go here...
