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

import copy


class OpenApiSpecParser:
    def __init__(self, spec):
        self.full_spec = copy.deepcopy(spec)
        self.resolved_spec = self._resolve_all(self.full_spec)

    def _get_referenced_data(self, ref_path: str) -> dict:
        if not ref_path.startswith("#/"):
            return {"error": f"External reference {ref_path} not supported"}

        parts = ref_path.lstrip("#/").split("/")
        current = self.full_spec

        for part in parts:
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, list) and part.isdigit():
                current = current[int(part)]
            else:
                raise KeyError(f"Could not resolve pointer part: {part}")

        return current

    def _resolve_all(self, data: any, seen_refs=None) -> any:
        if seen_refs is None:
            seen_refs = set()

        if isinstance(data, dict):
            if "$ref" in data:
                ref = data["$ref"]
                if ref in seen_refs:
                    return data  # Stop recursion on circularity

                # Update seen_refs for this branch
                new_seen = seen_refs | {ref}
                resolved_data = self._get_referenced_data(ref)
                return self._resolve_all(resolved_data, new_seen)

            # Recurse into dictionary values
            return {k: self._resolve_all(v, seen_refs) for k, v in data.items()}

        elif isinstance(data, list):
            # Recurse into list items
            return [self._resolve_all(item, seen_refs) for item in data]

        return data

    def get_path_request_schema(self, path: str, method: str) -> dict:
        method = method.lower()
        op = self.resolved_spec.get("paths", {}).get(path, {}).get(method, {})
        return op["requestBody"]["content"]["application/json"]["schema"]

    def get_path_response_schema(self, path: str, method: str) -> dict:
        method = method.lower()
        op = self.resolved_spec.get("paths", {}).get(path, {}).get(method, {})
        return op["responses"]["200"]["content"]["application/json"]["schema"]
