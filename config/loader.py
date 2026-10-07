from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - used only when requirements are incomplete
    yaml = None


DEFAULT_SETTINGS = Path(__file__).with_name("settings.yaml")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_yaml(settings_path: Path) -> dict[str, Any]:
    with settings_path.open("r", encoding="utf-8") as f:
        if yaml is not None:
            return yaml.safe_load(f) or {}
        return _load_simple_yaml(f.read())


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = _deep_merge(existing, value)
        else:
            result[key] = value
    return result


def load_settings(
    path: str | Path | None = None, *, use_local_override: bool = True
) -> dict[str, Any]:
    settings_path = Path(path) if path else DEFAULT_SETTINGS
    data = _read_yaml(settings_path)

    if use_local_override:
        local_path = settings_path.with_name(
            f"{settings_path.stem}.local{settings_path.suffix}"
        )
        if local_path.exists():
            data = _deep_merge(data, _read_yaml(local_path))

    return data


def get_nested(settings: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = settings
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _load_simple_yaml(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not raw_line.startswith(" "):
            key = line.rstrip(":")
            result[key] = {}
            current_section = result[key]
            continue
        if current_section is None or ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        current_section[key] = _parse_scalar(value)
    return result


def _parse_scalar(value: str) -> Any:
    if value in {"", "null", "None"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item.strip()) for item in inner.split(",")]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")
