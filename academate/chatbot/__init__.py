"""
Chatbot module for Academate.
"""

# Import selfrag classes for backward compatibility
from academate.chatbot.selfrag import ChatbotApp, clean_answer, remap_source_citations

# Import the new implementation
from academate.chatbot.chatbot_implementation import AcademateChat

__all__ = [
    "ChatbotApp",
    "clean_answer",
    "remap_source_citations",
    "AcademateChat"  # Add the new implementation
]