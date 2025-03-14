"""
Variant implementations of Academate.
"""

from academate.variants.single_db import AcademateSingleDB
from academate.variants.numeric import NumericAcademate

__all__ = [
    "AcademateSingleDB",
    "NumericAcademate"
]