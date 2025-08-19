import re
from typing import Optional, Tuple

import chainlit as cl


class ResponseParser:
    """Enhanced response parser with better UX for thinking process"""

    @staticmethod
    def extract_think_and_answer(response: str) -> Tuple[Optional[str], str]:
        """Extract thinking process and final answer"""
        if not response:
            return None, response

        # Pattern to match <think>...</think> tags
        think_pattern = r"<think>(.*?)</think>"

        # Extract thinking content
        think_match = re.search(
            think_pattern,
            response,
            re.DOTALL | re.IGNORECASE)
        thinking_process = think_match.group(
            1).strip() if think_match else None

        # Remove think tags to get final answer
        final_answer = re.sub(
            think_pattern,
            "",
            response,
            flags=re.DOTALL | re.IGNORECASE).strip()

        return thinking_process, final_answer

    @staticmethod
    def extract_content(response) -> str:
        """Extract content from various response types"""
        if hasattr(response, "content"):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)


async def display_response_with_thinking_step(
        response_content: str,
        step_name: str = "Analysis"):
    """Display response with thinking as a separate chainlit step"""

    thinking, answer = ResponseParser.extract_think_and_answer(
        response_content)

    if thinking:
        step_title = (
            f"🧠 {step_name} - Reasoning" if "Reasoning" not in step_name else f"🧠 {step_name}")
        async with cl.Step(name=step_title, type="tool") as thinking_step:
            thinking_step.output = thinking

        # Then show the main answer
        await cl.Message(content=answer).send()
    else:
        # No thinking process, just show answer
        await cl.Message(content=answer).send()
