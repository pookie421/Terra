"""Reconstruct typed source snapshots from local JSON (no executable deserialization)."""
import dataclasses
from typing import Union, get_args, get_origin, get_type_hints

def hydrate(annotation, value):
    if value is None:
        return None
    origin, arguments = get_origin(annotation), get_args(annotation)
    if origin is Union:
        candidates = [item for item in arguments if item is not type(None)]
        if len(candidates) != 1:
            raise ValueError('Snapshot replay requires unambiguous optional types')
        return hydrate(candidates[0], value)
    if dataclasses.is_dataclass(annotation):
        hints = get_type_hints(annotation)
        return annotation(**{key: hydrate(hints[key], item) for key, item in value.items()})
    if origin is list:
        return [hydrate(arguments[0], item) for item in value]
    if origin is dict:
        return {key: hydrate(arguments[1], item) for key, item in value.items()}
    if origin is tuple:
        return tuple(value)
    return value
