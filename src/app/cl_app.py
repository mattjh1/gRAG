import asyncio
import os
import sys
import uuid
from typing import Any, Dict, List, Optional

import chainlit as cl
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from loguru import logger

from app.util import get_rag_chain, get_summary_chain
from app.util.agent import AgenticGraphRAG
from app.util.chains import ConversationState
from app.util.planner import QueryPlanner
from app.util.response import ResponseParser, display_response_with_thinking_step
from app.util.validators import ResponseValidator

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class SessionKeyError(Exception):
    """Custom exception for session key errors."""

    pass


def set_session(key: str, value: Any) -> None:
    """Set a value in the user session."""
    return cl.user_session.set(key, value)


def get_session(key: str) -> Any:
    """Get a value from the user session."""
    default_ = object()
    value = cl.user_session.get(key, default=default_)
    if value == default_:
        raise SessionKeyError(
            f"Object '{key}' not found in `cl.user_session`.")
    return value


async def get_domain_info():
    """Get simple domain information without graph queries."""
    return {
        "domain_name": "Dracula by Bram Stoker - Literary Analysis",
        "entity_count": "500+",
        "relationship_count": "1000+",
        "top_entity_types": ["Character", "Theme", "Location"],
        "example_queries": """
• "Analyze Jonathan Harker's psychological evolution throughout the novel"
• "How does the theme of modernity vs. ancient evil develop through character interactions?"
• "Trace the symbolic significance of blood across different character perspectives"
• "Compare narrative techniques in different character journal entries"
        """.strip(),
    }


@cl.on_chat_start
async def on_chat_start():
    # Get domain information from the knowledge graph
    domain_info = await get_domain_info()

    welcome_message = f"""
#  Agentic Knowledge Explorer

I'm an AI agent that deeply understands your knowledge domain through an advanced knowledge graph.
I can:

 **Plan multi-step analysis** across entities, themes, and concepts
 **Follow relationship paths** through interconnected information
✅ **Validate my reasoning** and provide confidence scores
 **Show my thinking process** as I work through complex questions

## Your Knowledge Domain: {domain_info["domain_name"]}
**Entities:** {domain_info["entity_count"]} | **Relationships:** {domain_info["relationship_count"]}
**Key Entity Types:** {", ".join(domain_info["top_entity_types"])}

## Example Complex Queries:
{domain_info["example_queries"]}

*I'll show you my step-by-step reasoning as I work through your questions!*
"""

    await cl.Message(content=welcome_message).send()

    # Initialize components with modern conversation state
    conversation_state = ConversationState()
    rag_chain = get_rag_chain()
    summary_chain = get_summary_chain()

    # Initialize agentic components
    query_planner = QueryPlanner()
    response_validator = ResponseValidator()
    agentic_rag = AgenticGraphRAG(
        base_rag_chain=rag_chain,
        planner=query_planner,
        validator=response_validator,
        memory=conversation_state,  # Pass the new conversation state
    )

    set_session("conversation_state", conversation_state)
    set_session("rag_chain", rag_chain)
    set_session("summary_chain", summary_chain)
    set_session("agentic_rag", agentic_rag)
    set_session("query_planner", query_planner)


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages with agentic processing."""

    # Get session components
    conversation_state: ConversationState = get_session("conversation_state")
    agentic_rag: AgenticGraphRAG = get_session("agentic_rag")

    # Step 1: Analyze query complexity
    complexity_analysis = await analyze_query_complexity(message.content, conversation_state)

    if complexity_analysis["is_simple"]:
        # Simple query - use direct RAG
        await handle_simple_query(message, conversation_state, get_session("rag_chain"))
    else:
        # Complex query - use agentic approach
        await handle_complex_query(message, agentic_rag, complexity_analysis)


async def analyze_query_complexity(
        query: str,
        conversation_state: ConversationState) -> Dict:
    """Determine if query needs agentic processing."""

    # Generic complexity indicators that work across domains
    complexity_indicators = [
        "trace",
        "compare",
        "analyze",
        "evolution",
        "cascading",
        "influence",
        "throughout",
        "across",
        "connect",
        "relationship",
        "effects",
        "impact",
        "how does",
        "what happens when",
        "explain the connection",
        "contrast",
        "similarities",
        "differences",
        "patterns",
        "trends",
        "correlations",
    ]

    has_complexity_words = any(word in query.lower()
                               for word in complexity_indicators)
    has_multiple_concepts = len(query.split()) > 10
    has_conversation_context = len(conversation_state.get_messages()) > 0

    return {
        "is_simple": not (
            has_complexity_words or has_multiple_concepts),
        "needs_planning": has_complexity_words,
        "needs_multi_step": any(
            word in query.lower() for word in [
                "trace",
                "throughout",
                "evolution",
                "cascading"]),
        "needs_comparison": any(
            word in query.lower() for word in [
                "compare",
                "contrast",
                "similarities",
                "differences"]),
        "needs_relationship_analysis": any(
            word in query.lower() for word in [
                "connect",
                "relationship",
                "influence",
                "impact"]),
        "context_dependent": has_conversation_context,
    }


async def handle_simple_query(
        message: cl.Message,
        conversation_state: ConversationState,
        rag_chain: Runnable):
    """Handle simple queries with proper streaming that separates thinking from answer."""
    query = {"question": message.content}
    chat_messages = conversation_state.get_messages()
    if chat_messages:
        query["chat_history"] = chat_messages

    full_response = ""
    current_content = ""
    thinking_step = None
    main_msg = None
    in_think_tags = False

    async for chunk in rag_chain.astream(
        query,
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        full_response += chunk
        current_content += chunk

        # Check if we're entering think tags
        if "<think>" in current_content and not in_think_tags:
            in_think_tags = True
            thinking_step = cl.Step(name="🧠 Reasoning...", type="tool")
            await thinking_step.__aenter__()
            # Remove the <think> tag from display
            current_content = current_content.replace("<think>", "")
            continue

        # Check if we're exiting think tags
        if "</think>" in current_content and in_think_tags:
            in_think_tags = False
            # Extract thinking content and close step
            think_content = current_content.replace("</think>", "")
            thinking_step.output = think_content
            await thinking_step.__aexit__(None, None, None)

            # Start main message for the answer
            main_msg = cl.Message(content="")
            current_content = ""  # Reset for answer content
            continue

        # Stream to appropriate destination
        if in_think_tags and thinking_step:
            # Update thinking step (note: chainlit steps don't support
            # streaming)
            thinking_step.output = current_content
        elif main_msg:
            # Stream to main answer
            await main_msg.stream_token(chunk)
        elif not in_think_tags and not main_msg:
            # No thinking tags detected, create main message if not exists
            if not main_msg:
                main_msg = cl.Message(content="")
            await main_msg.stream_token(chunk)

    # Finalize main message
    if main_msg:
        await main_msg.update()

    # Extract clean answer for conversation state
    _, clean_answer = ResponseParser.extract_think_and_answer(full_response)

    # Save clean answer to conversation state
    conversation_state.save_context(
        inputs={
            "human": message.content}, outputs={
            "ai": clean_answer})


# Updated complex query handler for consistency
async def handle_complex_query(
        message: cl.Message,
        agentic_rag: AgenticGraphRAG,
        complexity_analysis: Dict):
    """Handle complex queries with agentic processing."""

    try:
        # Step 1: Show planning phase
        planning_msg = cl.Message(content="🎯 **Planning my approach...**")
        await planning_msg.send()

        # Generate execution plan
        plan = await agentic_rag.create_execution_plan(message.content, complexity_analysis)

        # Update planning message with the plan
        plan_text = "## 📋 My Analysis Plan\n\n"
        for i, step in enumerate(plan.steps, 1):
            plan_text += f"**Step {i}:** {step.description}\n"
        plan_text += f"\n*Estimated confidence: {plan.confidence_estimate}%*"

        planning_msg.content = plan_text
        await planning_msg.update()

        # Step 2: Execute plan with enhanced step display
        results = []

        for i, step in enumerate(plan.steps, 1):
            # Each step gets its own chainlit step for better UX
            step_title = f"Step {i}: {step.description}"
            step_context = cl.Step(name=step_title, type="tool")
            async with step_context as step_display:
                try:
                    # Execute step
                    step_result = await agentic_rag.execute_step(step, message.content)
                    results.append(step_result)

                    # Handle step result safely - extract text content
                    if hasattr(step_result, "summary") and step_result.summary:
                        result_text = str(step_result.summary)
                    elif hasattr(step_result, "data") and step_result.data:
                        # Try to extract meaningful text from data
                        data = step_result.data
                        if isinstance(data, dict):
                            if "result" in data:
                                result_text = str(data["result"])
                            elif "synthesis" in data:
                                result_text = str(data["synthesis"])
                            elif "answer" in data:
                                result_text = str(data["answer"])
                            else:
                                # Join all non-empty string values
                                values = [
                                    str(v) for v in data.values() if v and str(v).strip()]
                                result_text = " ".join(
                                    values) if values else str(data)
                        else:
                            result_text = str(data)
                    else:
                        result_text = f"Step {i} completed successfully"

                    # Parse step result for thinking vs summary
                    thinking, summary = ResponseParser.extract_think_and_answer(
                        result_text)

                    if thinking:
                        step_display.output = (
                            f"**🧠 Reasoning:**\n{thinking}\n\n**📝 Result:**\n{summary}")
                    else:
                        step_display.output = summary or result_text

                    # Don't store the step_result object to avoid serialization issues
                    # Instead, store just basic metadata as a simple dict
                    step_display.generation = {
                        "success": getattr(
                            step_result, "success", True), "confidence": getattr(
                            step_result, "confidence", 0), "execution_time": getattr(
                            step_result, "execution_time", 0), "step_type": getattr(
                            step, "query_type", "unknown"), }

                except Exception as e:
                    import traceback

                    error_details = f"❌ **Error in Step {i}:** {
                        str(e)
                    }\n\n**Debug Info:**\n```\n{traceback.format_exc()}\n```"
                    step_display.output = error_details
                    print(f"Step execution error: {e}")  # For server logs
                    # Add a None result to maintain step count
                    results.append(None)
                    continue

        # Step 3: Synthesize results with enhanced display
        step_context = cl.Step(name="🔄 Synthesizing Results", type="tool")
        async with step_context as synthesis_step:
            try:
                # Filter successful results (remove None values from failed
                # steps)
                successful_results = [
                    r for r in results if r is not None and getattr(
                        r, "success", False)]

                if not successful_results:
                    synthesis_step.output = (
                        "❌ No successful steps to synthesize - all steps failed"
                    )
                    raise RuntimeError(
                        "No successful step results to synthesize")

                synthesis_step.output = f"🔄 Processing {
                    len(successful_results)} successful results..."

                final_result = await agentic_rag.synthesize_results(
                    successful_results, message.content
                )
                synthesis_step.output = f"✅ Successfully combined {
                    len(successful_results)
                } findings into comprehensive analysis"

            except Exception as e:
                import traceback

                error_details = f"❌ Synthesis failed: {
                    str(e)
                }\n\n**Debug Info:**\n```\n{traceback.format_exc()}\n```"
                synthesis_step.output = error_details
                print(f"Synthesis error: {e}")  # For server logs
                raise

        # Step 4: Display final result with thinking separation
        # Safely extract the answer text
        if hasattr(final_result, "answer"):
            answer_text = str(final_result.answer)
        else:
            answer_text = str(final_result)

        await display_response_with_thinking_step(answer_text, "Final Analysis")

        # Step 5: Show metadata in a clean format
        metadata_content = f"""## 📊 Analysis Summary

**🎯 Confidence Level:** {final_result.confidence}%
**📚 Sources Consulted:** {len(final_result.sources)} documents
**⏱️ Steps Completed:** {len([r for r in results if r.success])}/{len(results)}
"""

        # Add limitations if present
        if final_result.limitations:
            _, clean_limitations = ResponseParser.extract_think_and_answer(
                final_result.limitations)
            metadata_content += f"\n**⚠️ Analysis Limitations:**\n{clean_limitations}"

        # Add collapsible sources section
        if final_result.sources:
            metadata_content += f"""

<details>
<summary><strong>📚 View Detailed Sources</strong></summary>

{chr(10).join(f"• {source}" for source in final_result.sources)}

</details>
"""

        await cl.Message(content=metadata_content).send()

        # Save clean answer to conversation state
        _, clean_answer = ResponseParser.extract_think_and_answer(
            final_result.answer)
        agentic_rag.memory.save_context(
            inputs={"human": message.content}, outputs={"ai": clean_answer}
        )

    except Exception as e:
        # Enhanced error handling
        error_msg = cl.Message(
            content=f"""## ❌ Complex Analysis Failed

**Error:** {str(e)}

**🔄 Falling back to simple search...**

*This usually happens when the query is too complex for the current model or when there are connectivity issues.*
"""
        )
        await error_msg.send()

        # Fallback to simple query
        try:
            await handle_simple_query(message, agentic_rag.memory, agentic_rag.base_rag_chain)
        except Exception as fallback_error:
            await cl.Message(content=f"❌ **Fallback also failed:** {str(fallback_error)}").send()


def infer_domain_name(
        entity_types: List[str],
        sample_entities: List[str]) -> str:
    """Infer domain name from entity patterns."""

    # Check for common domain patterns - reduce return statements
    if any(t in ["Person", "Character"] for t in entity_types):
        return (
            "Literary Analysis (Dracula)"
            if any("dracula" in str(e).lower() for e in sample_entities)
            else "Character/Person Analysis"
        )

    domain_type_map = {
        ("Company", "Organization"): "Business/Organization Analysis",
        ("Gene", "Protein", "Disease"): "Biomedical Knowledge",
        ("Concept", "Topic", "Theme"): "Conceptual Knowledge",
        ("Document", "Paper", "Article"): "Document Analysis",
    }

    for type_group, domain_name in domain_type_map.items():
        if any(t in type_group for t in entity_types):
            return domain_name

    return "Knowledge Domain"


def generate_example_queries(
        entity_types: List[str],
        sample_entities: List[str]) -> str:
    """Generate domain-appropriate example queries."""

    examples = []

    # Generic examples that work for any domain
    if sample_entities:
        entity_example = sample_entities[0] if sample_entities else "key entities"
        examples.append(
            f'- **"Trace the relationships involving {entity_example}"**')

    if len(entity_types) >= 2:
        examples.append(
            f'- **"Compare {entity_types[0]} and {
                entity_types[1]
            } entities in the knowledge graph"**'
        )

    # Domain-specific examples
    if "Person" in entity_types or "Character" in entity_types:
        examples.extend(
            [
                '- **"Analyze the influence patterns between characters"**',
                '- **"What are the cascading effects when key characters interact?"**',
            ]
        )
    elif "Company" in entity_types or "Organization" in entity_types:
        examples.extend(
            [
                '- **"How do organizational relationships impact business outcomes?"**',
                '- **"Trace the network effects between companies"**',
            ])
    elif "Concept" in entity_types:
        examples.extend(
            [
                '- **"How do these concepts connect to form larger themes?"**',
                '- **"What are the conceptual dependencies in this domain?"**',
            ]
        )
    else:
        examples.extend(
            [
                '- **"Analyze the relationship patterns in your knowledge graph"**',
                '- **"How do different entities influence each other?"**',
            ])

    return "\n".join(examples)
