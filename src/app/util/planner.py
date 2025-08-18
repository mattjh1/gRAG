from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio

from langchain_core.messages import BaseMessage
from app.core.config import LLMSettings
from util.llm import get_ollama_instance
import json


@dataclass
class ExecutionStep:
    """Single step in query execution plan."""

    step_id: str
    description: str
    query_type: str  # "entity_search", "relationship_traverse", "synthesis"
    parameters: Dict[str, Any]
    dependencies: List[str]  # IDs of steps this depends on
    expected_confidence: float


@dataclass
class ExecutionPlan:
    """Complete execution plan for a complex query."""

    steps: List[ExecutionStep]
    confidence_estimate: float
    estimated_time: float
    fallback_strategy: Optional[str] = None


@dataclass
class StepResult:
    """Result of executing a single step."""

    step_id: str
    success: bool
    data: Any
    confidence: float
    summary: str
    sources: List[str]
    execution_time: float


@dataclass
class FinalResult:
    """Final synthesized result."""

    answer: BaseMessage
    confidence: float
    sources: List[str]
    limitations: Optional[str] = None
    reasoning_chain: Optional[List[str]] = None


class QueryPlanner:
    """Plans multi-step execution for complex queries."""

    def __init__(self, settings: Optional[LLMSettings] = None):
        self.llm = get_ollama_instance(settings)

    async def create_execution_plan(
        self, query: str, complexity_analysis: Dict
    ) -> ExecutionPlan:
        """Create a step-by-step execution plan."""

        planning_prompt = f"""
        You are a query planner for a GraphRAG system with a generic knowledge graph.
        
        Query: {query}
        Complexity Analysis: {json.dumps(complexity_analysis, indent=2)}
        
        Create a step-by-step execution plan. Consider:
        - Entity extraction and search from the query
        - Relationship traversal in the knowledge graph  
        - Multi-hop reasoning across connected concepts
        - Synthesis of findings from multiple sources
        
        Return a JSON plan with steps, each having:
        - step_id: unique identifier
        - description: what this step does
        - query_type: one of ["entity_search", "relationship_traverse", "synthesis", "validation"]
        - parameters: specific parameters for this step
        - dependencies: list of step_ids this depends on
        - expected_confidence: 0-100 estimate
        
        Example format:
        {{
            "steps": [
                {{
                    "step_id": "extract_entities",
                    "description": "Extract key entities and concepts from the query",
                    "query_type": "entity_search", 
                    "parameters": {{"entities": ["concept1", "concept2"]}},
                    "dependencies": [],
                    "expected_confidence": 90
                }}
            ],
            "confidence_estimate": 85,
            "estimated_time": 10.5
        }}
        """

        # Run blocking LLM call in a background thread
        response = await asyncio.to_thread(self.llm.invoke, planning_prompt)

        plan_json = json.loads(str(response))  # make sure it's stringified

        steps = [
            ExecutionStep(
                step_id=step["step_id"],
                description=step["description"],
                query_type=step["query_type"],
                parameters=step["parameters"],
                dependencies=step["dependencies"],
                expected_confidence=step["expected_confidence"],
            )
            for step in plan_json["steps"]
        ]

        return ExecutionPlan(
            steps=steps,
            confidence_estimate=plan_json["confidence_estimate"],
            estimated_time=plan_json["estimated_time"],
        )
