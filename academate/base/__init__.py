"""
Base classes for Academate components.
"""

from academate.base.core import AcademateBase
from academate.base.screener import AbstractScreener
from academate.base.embedder import AbstractEmbedder
from academate.base.analyzer import AbstractAnalyzer

__all__ = [
    "AcademateBase",
    "AbstractScreener",
    "AbstractEmbedder",
    "AbstractAnalyzer"
]