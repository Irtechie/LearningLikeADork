# Purpose: Module for catalog.
# Created: 2026-01-07
# Author: MWR

import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Dict, Iterable


CONFIG_PATH = Path(__file__).resolve().parent / "models.json"


class ModelNames:
    FAST = "fast"
    REASONING = "reasoning"
    PERFORMANCE = "performance"
    FULL = "full"
    SMALL = "small"
    TINY = "tiny"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: str
    adapter: str

    def adapter_cls(self):
        module_path, class_name = self.adapter.rsplit(".", 1)
        module = import_module(module_path)
        return getattr(module, class_name)

    def build_adapter(self, max_new_tokens: int = 128, model_path: str | None = None):
        adapter_cls = self.adapter_cls()
        return adapter_cls(
            model_path=model_path or self.path,
            max_new_tokens=max_new_tokens,
        )


class ModelCatalog:
    _cache: Dict[str, ModelSpec] | None = None

    @classmethod
    def _load(cls) -> Dict[str, ModelSpec]:
        if cls._cache is not None:
            return cls._cache
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cls._cache = {
            name: ModelSpec(name=name, path=cfg["path"], adapter=cfg["adapter"])
            for name, cfg in data.items()
        }
        return cls._cache

    @classmethod
    def get(cls, name: str) -> ModelSpec:
        items = cls._load()
        if name not in items:
            raise KeyError(f"Unknown model name: {name}")
        return items[name]

    @classmethod
    def names(cls) -> Iterable[str]:
        return sorted(cls._load().keys())
