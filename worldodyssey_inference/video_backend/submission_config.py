from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


ConfigDict = dict[str, Any]
MISSING = object()


def load_yaml_config(path: Path | None) -> ConfigDict:
    if path is None:
        return {}

    resolved = path.expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Submission config not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Submission config must be a file: {resolved}")

    try:
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML submission config {resolved}: {exc}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Submission config root must be a mapping: {resolved}")
    return payload


def parse_dotted_overrides(expressions: Iterable[str]) -> ConfigDict:
    config: ConfigDict = {}
    for expression in expressions:
        path, value = parse_dotted_override(expression)
        set_dotted_value(config, path, value)
    return config


def parse_dotted_override(expression: str) -> tuple[str, Any]:
    path, separator, raw_value = expression.partition("=")
    if not separator:
        raise ValueError(f"Expected override in dotted.path=value form, got: {expression!r}")
    path = path.strip()
    if not path:
        raise ValueError(f"Override path cannot be empty: {expression!r}")
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML value in override {expression!r}: {exc}") from exc
    return path, value


def get_dotted_value(config: ConfigDict, path: str, default: Any = MISSING) -> Any:
    current: Any = config
    for part in split_dotted_path(path):
        if isinstance(current, dict):
            if part not in current:
                if default is MISSING:
                    raise KeyError(path)
                return default
            current = current[part]
            continue
        if isinstance(current, list):
            if not _is_list_index(part):
                if default is MISSING:
                    raise KeyError(path)
                return default
            item_index = int(part)
            if item_index >= len(current):
                if default is MISSING:
                    raise KeyError(path)
                return default
            current = current[item_index]
            continue
        if default is MISSING:
            raise KeyError(path)
        return default
    return current


def set_dotted_value(config: ConfigDict, path: str, value: Any) -> None:
    parts = split_dotted_path(path)
    current: Any = config
    for index, part in enumerate(parts[:-1]):
        current = _descend_or_create(current, part, parts[index + 1], path)
    _assign_dotted_value(current, parts[-1], value, path)


def _descend_or_create(container: Any, part: str, next_part: str, path: str) -> Any:
    if isinstance(container, dict):
        existing = container.get(part)
        if existing is None:
            child: Any = [] if _is_list_index(next_part) else {}
            container[part] = child
            return child
        if not isinstance(existing, (dict, list)):
            raise ValueError(f"Cannot set {path!r}: {part!r} is not a mapping or list.")
        return existing

    if isinstance(container, list):
        item_index = _parse_list_index(part, path)
        _ensure_list_index(container, item_index)
        existing = container[item_index]
        if existing is None:
            child = [] if _is_list_index(next_part) else {}
            container[item_index] = child
            return child
        if not isinstance(existing, (dict, list)):
            raise ValueError(f"Cannot set {path!r}: list item {part!r} is not a mapping or list.")
        return existing

    raise ValueError(f"Cannot set {path!r}: parent is not a mapping or list.")


def _assign_dotted_value(container: Any, part: str, value: Any, path: str) -> None:
    if isinstance(container, dict):
        container[part] = value
        return
    if isinstance(container, list):
        item_index = _parse_list_index(part, path)
        _ensure_list_index(container, item_index)
        container[item_index] = value
        return
    raise ValueError(f"Cannot set {path!r}: parent is not a mapping or list.")


def _parse_list_index(part: str, path: str) -> int:
    if not _is_list_index(part):
        raise ValueError(f"Cannot set {path!r}: expected a non-negative list index, got {part!r}.")
    return int(part)


def _ensure_list_index(values: list[Any], index: int) -> None:
    while len(values) <= index:
        values.append(None)


def _is_list_index(part: str) -> bool:
    return part.isdigit()


def split_dotted_path(path: str) -> list[str]:
    parts = [part.strip() for part in path.split(".")]
    if not parts or any(not part for part in parts):
        raise ValueError(f"Invalid dotted path: {path!r}")
    return parts


def deep_merge(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        current_value = result.get(key)
        if isinstance(current_value, dict) and isinstance(value, dict):
            result[key] = deep_merge(current_value, value)
        elif isinstance(current_value, list) and isinstance(value, list):
            result[key] = _deep_merge_lists(current_value, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _deep_merge_lists(base: list[Any], override: list[Any]) -> list[Any]:
    result = copy.deepcopy(base)
    for index, value in enumerate(override):
        if value is None:
            continue
        if index >= len(result):
            result.append(copy.deepcopy(value))
            continue
        current_value = result[index]
        if isinstance(current_value, dict) and isinstance(value, dict):
            result[index] = deep_merge(current_value, value)
        else:
            result[index] = copy.deepcopy(value)
    return result
