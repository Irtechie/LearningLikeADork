# Purpose: Backward-compatible shim for catalog imports.
# Created: 2026-01-08
# Author: MWR

"""Backward-compatible shim for catalog imports."""

from .catalog import ModelCatalog, ModelNames, ModelSpec

__all__ = ["ModelCatalog", "ModelNames", "ModelSpec"]
