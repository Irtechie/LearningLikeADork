# Purpose: Local model adapter.
# Created: 2026-01-08
# Author: MWR

from .local_hf import LocalHFAdapter as _BaseLocalHFAdapter
from models.catalog import ModelCatalog, ModelNames


class LocalHFAdapter(_BaseLocalHFAdapter):
    def __init__(self, model_path: str | None = None, max_new_tokens: int = 128):
        if model_path is None:
            model_path = ModelCatalog.get(ModelNames.FULL).path
        super().__init__(model_path=model_path, max_new_tokens=max_new_tokens)
