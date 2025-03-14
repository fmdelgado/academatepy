"""
Embedder implementations for Academate.
"""

from academate.embedders.literature import LiteratureEmbedder
from academate.embedders.pdf import PDFEmbedder

__all__ = [
    "LiteratureEmbedder",
    "PDFEmbedder"
]