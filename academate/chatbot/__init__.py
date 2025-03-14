"""
Chatbot module for Academate.
"""

from academate.chatbot.selfrag import ChatbotApp, clean_answer, remap_source_citations

__all__ = [
    "ChatbotApp",
    "clean_answer",
    "remap_source_citations"
]