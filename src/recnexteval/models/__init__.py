"""Deprecated: Use recnexteval.core instead."""

import warnings

from ..core import BaseModel, ParamMixin

warnings.warn(
    "recnexteval.models is deprecated, use recnexteval.core instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["BaseModel", "ParamMixin"]
