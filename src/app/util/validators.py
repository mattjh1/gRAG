from typing import Any, Dict, Optional

from util.llm import get_ollama_instance

from app.core.config import LLMSettings
from app.util.planner import FinalResult, StepResult


# app/util/validators.py
class ResponseValidator:
    """Validates quality and completeness of agentic responses."""

    def __init__(self, settings: Optional[LLMSettings] = None):
        self.llm = get_ollama_instance(settings)

    async def validate_step_result(self, result: StepResult) -> float:
        """Validate individual step result quality."""

        # Quick heuristic validation
        confidence_factors = []

        # Check if result has content
        if result.data and len(str(result.data)) > 50:
            confidence_factors.append(0.3)

        # Check if sources are present
        if result.sources and len(result.sources) > 0:
            confidence_factors.append(0.2)

        # Check execution success
        if result.success:
            confidence_factors.append(0.3)

        # Use reported confidence
        confidence_factors.append(result.confidence / 100 * 0.2)

        return sum(confidence_factors) * 100

    async def validate_final_result(
        self, result: FinalResult, original_query: str
    ) -> Dict[str, Any]:
        """Validate the final synthesized result."""

        validation_prompt = f"""
        Evaluate this response to the knowledge query:

        Original Query: {original_query}
        Response: {result.answer}

        Rate 1-10 on:
        - Relevance: Does it directly address the query?
        - Completeness: Does it fully address all aspects of the query?
        - Coherence: Is the reasoning logical and well-structured?
        - Evidence: Are there sufficient supporting details?
        - Accuracy: Does the information seem factually consistent?

        Return JSON: {{"relevance": 8, "completeness": 7, "coherence": 9, "evidence": 6, "accuracy": 8, "overall": 7.6}}
        """

        # Mock validation for demo - in real implementation call LLM
        return {
            "relevance": 8.5,
            "completeness": 8.0,
            "coherence": 9.0,
            "evidence": 7.5,
            "accuracy": 8.2,
            "overall": 8.2,
            "needs_improvement": [],
        }
