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

from typing import Any, Dict, Type, Tuple
from pydantic import BaseModel, create_model, Field, ConfigDict


class SchemaParser:
    def __init__(self, datatypes: Dict[str, Any]):
        self.raw_schemas = datatypes
        self.models: Dict[str, Type[BaseModel]] = {}

    def _json_type_to_python(self, prop_def: Dict[str, Any]) -> Tuple[Any, Any]:
        """
        Maps JSON schema types to Python types and returns
        a tuple of (type, FieldInfo).
        """
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        # Handle References
        if "$ref" in prop_def:
            ref_name = prop_def["$ref"]
            return self.parse_type(ref_name), ...

        # Determine base type
        json_type = prop_def.get("type", "string")

        if json_type == "array":
            items_def = prop_def.get("items", {})
            item_type, _ = self._json_type_to_python(items_def)
            python_type = list[item_type]
        elif json_type == "object" and "properties" in prop_def:
            # Inline object definition, create a nested model
            # We use a generic name or something derived from field name if we had it.
            # But _json_type_to_python doesn't have field name.
            # Actually, the best way to handle inline objects is to give them a name.
            # For now, let's just create an anonymous model.
            nested_fields = {
                k: self._json_type_to_python(v)
                for k, v in prop_def.get("properties", {}).items()
            }
            python_type = create_model(
                "InlineModel", __config__=ConfigDict(extra="forbid"), **nested_fields
            )
        else:
            python_type = type_map.get(json_type, Any)

        # Build Field constraints
        field_kwargs = {}

        # Support maxLength -> max_length
        if "max_length" in prop_def:
            field_kwargs["max_length"] = prop_def["max_length"]

        # Support minLength -> min_length
        if "min_length" in prop_def:
            field_kwargs["min_length"] = prop_def["min_length"]

        # If we have constraints, return (type, Field(...)), else (type, ...)
        if field_kwargs:
            return python_type, Field(**field_kwargs)

        return python_type, ...

    def parse_type(self, type_name: str) -> Type[BaseModel]:
        """Recursively parses a schema definition into a Pydantic model."""
        if type_name in self.models:
            return self.models[type_name]

        schema = self.raw_schemas.get(type_name)
        if not schema:
            raise ValueError(f"Definition for '{type_name}' not found.")

        # Handle top-level $ref
        if "$ref" in schema:
            ref_name = schema["$ref"]
            ref_model = self.parse_type(ref_name)
            # Create a subclass or alias? Subclass with the new name is better for clarity.
            model = create_model(
                type_name,
                __base__=ref_model,
                __config__=ConfigDict(extra="forbid"),
            )
            self.models[type_name] = model
            return model

        properties = schema.get("properties", {})
        fields = {}

        for field_name, field_def in properties.items():
            # Now returns (type, FieldInfo)
            fields[field_name] = self._json_type_to_python(field_def)

        model = create_model(type_name, __config__=ConfigDict(extra="forbid"), **fields)
        self.models[type_name] = model
        return model

    def parse_all(self) -> Dict[str, Type[BaseModel]]:
        for type_name in self.raw_schemas:
            self.parse_type(type_name)
        return self.models
