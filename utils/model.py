from __future__ import annotations

from typing import Any


def model_to_dict(model: Any) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def model_to_json(model: Any, **kwargs: Any) -> str:
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json(**kwargs)
    return model.json(**kwargs)
