"""
Academate: Streamlining and supporting systematic review processes using LLMs.
"""

__version__ = "1.0.0"

# Import main variant implementations for easy access
from academate.variants.single_db import AcademateSingleDB
from academate.variants.numeric import NumericAcademate

# For backward compatibility
import warnings

def warn_legacy_import(name):
    warnings.warn(
        f"{name} is imported from a legacy location. Please update your imports to use "
        f"the new package structure.", DeprecationWarning, stacklevel=2
    )

# Expose key classes directly at the package level
__all__ = [
    "AcademateSingleDB",
    "NumericAcademate"
]