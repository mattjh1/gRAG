import asyncio
from typing import Dict, List

from langchain_core.runnables import Runnable

from app.util.chains import ConversationState, get_ner_chain
from app.util.llm import get_llm_instance
from app.util.planner import (
    ExecutionPlan,
    ExecutionStep,
    FinalResult,
    QueryPlanner,
    StepResult,
)
from app.util.retrievers import structured_retriever
from app.util.tools import graph_query_tool
from app.util.validators import ResponseValidator


class AgenticGraphRAG:
    """Main agentic wrapper for GraphRAG system."""

    def __init__(
        self,
        base_rag_chain: Runnable,
        planner: QueryPlanner,
        validator: ResponseValidator,
        memory: ConversationState,
    ):
        self.base_rag_chain = base_rag_chain
        self.planner = planner
        self.validator = validator
        self.memory = memory
        self.execution_history = []

    async def create_execution_plan(
        self, query: str, complexity_analysis: Dict
    ) -> ExecutionPlan:
        """Create execution plan for the query."""
        return await self.planner.create_execution_plan(query, complexity_analysis)

    async def execute_step(
        self, step: ExecutionStep, original_query: str
    ) -> StepResult:
        """Execute a single step in the plan."""

        start_time = asyncio.get_event_loop().time()

        try:
            if step.query_type == "entity_search":
                result_data = await self._execute_entity_search(step, original_query)
            elif step.query_type == "relationship_traverse":
                result_data = await self._execute_relationship_traverse(
                    step, original_query
                )
            elif step.query_type == "synthesis":
                result_data = await self._execute_synthesis(step, original_query)
            else:
                raise ValueError(f"Unknown query type: {step.query_type}")

            execution_time = asyncio.get_event_loop().time() - start_time

            result = StepResult(
                step_id=step.step_id,
                success=True,
                data=result_data,
                confidence=step.expected_confidence,
                summary=f"Successfully completed {step.description}",
                sources=result_data.get("sources", []),
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            result = StepResult(
                step_id=step.step_id,
                success=False,
                data={"error": str(e)},
                confidence=0,
                summary=f"Failed to execute {step.description}: {e}",
                sources=[],
                execution_time=execution_time,
            )

        self.execution_history.append(result)
        return result

    async def _execute_entity_search(self, step: ExecutionStep, query: str) -> Dict:
        """Execute entity search using structured retriever."""
        result = await asyncio.to_thread(structured_retriever, query)
        return {"result": result}

    async def _execute_relationship_traverse(self, query: str) -> Dict:
        """Execute relationship traversal in knowledge graph."""
        entities = await asyncio.to_thread(get_ner_chain().invoke, {"input": query})
        if len(entities.names) < 2:
            return {"result": "Not enough entities to find a path"}

        query = f"""MATCH p=allShortestPaths((a)-[*..5]-(b))
WHERE a.id = '{entities.names[0]}' AND b.id = '{entities.names[1]}'
RETURN p"""
        result = graph_query_tool(query)
        return {"result": result}

    async def _execute_synthesis(self, step: ExecutionStep, query: str) -> Dict:
        """Synthesize results from previous steps."""
        llm = get_llm_instance()
        previous_results = [r.data for r in self.execution_history if r.success]
        prompt = f"""Synthesize the following results:
{previous_results}

into a coherent answer for the query: {query}"""
        synthesis = await asyncio.to_thread(llm.invoke, prompt)
        return {"synthesis": synthesis}

    async def synthesize_results(
        self, results: List[StepResult], original_query: str
    ) -> FinalResult:
        """Create final synthesized response."""

        llm = get_llm_instance()
        successful_results = [r for r in results if r.success]
        all_sources = list(
            set([source for r in successful_results for source in r.sources])
        )

        prompt = f"""Based on the following analysis, provide a comprehensive answer to the user's query.

        Query: {original_query}

        Analysis:
        {[r.data for r in successful_results]}
        """

        synthesized_answer = await asyncio.to_thread(llm.invoke, prompt)

        # Calculate overall confidence
        avg_confidence = (
            sum(r.confidence for r in successful_results) / len(successful_results)
            if successful_results
            else 0
        )

        return FinalResult(
            answer=synthesized_answer,
            confidence=avg_confidence,
            sources=all_sources,
            reasoning_chain=[r.summary for r in successful_results],
        )
