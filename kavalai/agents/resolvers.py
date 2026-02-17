"""
Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Any
from pydantic import BaseModel


def resolve_path(obj: Any, path: str) -> Any:
    """
    Resolve a dotted path like 'input.user_message' from nested structures.
    Supports dicts, lists (by numeric index), BaseModel instances, and plain objects' attributes.
    Special segment 'length' returns len(current) when applicable.
    """
    if path is None or path == "":
        return obj
    parts = str(path).split(".")
    cur: Any = obj

    for part in parts:
        if cur is None:
            return None

        if part == "length":
            return len(cur) if hasattr(cur, "__len__") else None

        # index access for lists/tuples if the path segment is an int
        if isinstance(cur, (list, tuple)):
            if part.isdigit():
                idx = int(part)
                if 0 <= idx < len(cur):
                    cur = cur[idx]
                    continue
                return None
            # non-numeric access on list is not supported
            return None

        if isinstance(cur, dict):
            cur = cur.get(part)
            continue

        if isinstance(cur, BaseModel):
            # Prefer attribute access; pydantic v2 exposes fields directly
            if hasattr(cur, part):
                cur = getattr(cur, part)
                continue
            # As a fallback, try model_dump then dict access
            dumped = cur.model_dump()
            if part in dumped:
                cur = dumped[part]
                continue
            return None

        # generic attribute access on plain objects
        if hasattr(cur, part):
            cur = getattr(cur, part)
            continue

        return None

    return cur


def find_key_recursive(obj: Any, target: str) -> Any:
    """
    Recursively search for the first occurrence of a key/field named `target` in
    dicts, BaseModel instances, lists/tuples, or plain objects with attributes.
    Returns the first matching value found, or None.
    """
    if obj is None:
        return None

    # Direct hits
    if isinstance(obj, dict):
        if target in obj:
            return obj[target]
        # search values
        for v in obj.values():
            found = find_key_recursive(v, target)
            if found is not None:
                return found
        return None

    if isinstance(obj, BaseModel):
        if hasattr(obj, target):
            return getattr(obj, target)
        # search inside fields
        for name in obj.__class__.model_fields:
            val = getattr(obj, name)
            found = find_key_recursive(val, target)
            if found is not None:
                return found
        # fallback to dump
        dumped = obj.model_dump()
        for v in dumped.values():
            found = find_key_recursive(v, target)
            if found is not None:
                return found
        return None

    if isinstance(obj, (list, tuple)):
        for item in obj:
            found = find_key_recursive(item, target)
            if found is not None:
                return found
        return None

    # plain objects: check attribute directly, else try to recurse into public attributes
    if hasattr(obj, target):
        return getattr(obj, target)

    # Recurse into public attributes
    if hasattr(obj, "__dict__"):
        for val in obj.__dict__.values():
            found = find_key_recursive(val, target)
            if found is not None:
                return found

    return None
