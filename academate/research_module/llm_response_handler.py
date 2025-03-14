"""
LLM response handling utilities for standardizing interactions across different LLM types.
"""

import logging
from typing import Any, Dict, Optional, Union


class LLMResponseHandler:
    """
    Handles and normalizes responses from different LLM implementations.

    This class provides consistent extraction of text content from various
    LLM response formats, enabling more robust interactions regardless of
    the underlying LLM implementation.
    """

    def __init__(self, logger=None):
        """Initialize with optional logger."""
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def extract_content(self, response: Any) -> str:
        """
        Extract text content from an LLM response object.

        Args:
            response: The response object from any LLM

        Returns:
            str: The extracted text content

        This method handles different response formats from various LLMs:
        - String responses
        - Objects with a 'content' attribute (e.g. LangChain)
        - Dictionaries with a 'content' key
        - Other objects that can be converted to strings
        """
        try:
            # Handle string responses directly
            if isinstance(response, str):
                return response.strip()

            # Handle LangChain-style responses with .content attribute
            elif hasattr(response, 'content'):
                return response.content.strip()

            # Handle dictionary responses with 'content' key
            elif isinstance(response, dict) and 'content' in response:
                return response['content'].strip()

            # Handle OpenAI-style responses
            elif hasattr(response, 'choices') and hasattr(response.choices[0], 'message'):
                return response.choices[0].message.content.strip()

            # Handle Google Gemini responses
            elif hasattr(response, 'text'):
                return response.text.strip()

            # Try other attributes that might contain the content
            for attr in ['response', 'answer', 'output', 'result', 'generation']:
                if hasattr(response, attr):
                    value = getattr(response, attr)
                    if isinstance(value, str):
                        return value.strip()

            # Last resort: convert the whole response to string
            self.logger.warning("Unexpected response format. Converting to string.")
            return str(response).strip()

        except Exception as e:
            self.logger.error(f"Error extracting content from response: {e}")
            return ""

    def format_prompt(self, prompt_template: str, model_type: str = "general") -> str:
        """
        Format a prompt template based on the LLM type.

        Args:
            prompt_template: Base prompt template string
            model_type: Type of model ("openai", "google", "anthropic", etc.)

        Returns:
            str: Formatted prompt suitable for the specified model
        """
        # Default is to return the template unchanged
        formatted_prompt = prompt_template

        # Apply model-specific formatting
        model_type = model_type.lower()

        if model_type == "google":
            # Google models might need different marker formatting
            formatted_prompt = prompt_template.replace('<START_QUERY>', '```\n')
            formatted_prompt = formatted_prompt.replace('<END_QUERY>', '\n```')

        elif model_type == "anthropic":
            # Claude might benefit from explicit XML-style tags
            formatted_prompt = formatted_prompt.replace('<START_QUERY>', '<query>')
            formatted_prompt = formatted_prompt.replace('<END_QUERY>', '</query>')

        return formatted_prompt


def detect_llm_type(llm: Any) -> str:
    """
    Detect the type of LLM from the object.

    Args:
        llm: An LLM instance

    Returns:
        str: Detected LLM type ("openai", "google", "anthropic", "general")
    """
    if llm is None:
        return "general"

    # Get the module or class name
    module_name = getattr(llm, "__module__", "")
    class_name = llm.__class__.__name__
    full_name = f"{module_name}.{class_name}".lower()

    # Check for known LLM types
    if any(name in full_name for name in ["openai", "gpt"]):
        return "openai"
    elif any(name in full_name for name in ["google", "gemini", "palm", "bard"]):
        return "google"
    elif any(name in full_name for name in ["anthropic", "claude"]):
        return "anthropic"
    elif any(name in full_name for name in ["cohere"]):
        return "cohere"
    elif any(name in full_name for name in ["huggingface", "hf"]):
        return "huggingface"
    elif any(name in full_name for name in ["llama", "meta"]):
        return "meta"
    elif hasattr(llm, "_llm_type"):
        # LangChain models have this attribute
        return llm._llm_type.lower()

    # Default
    return "general"