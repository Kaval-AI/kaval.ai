from typing import Any, Dict, Type, Tuple
from pydantic import BaseModel, create_model, Field


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
            model = create_model(type_name, __base__=ref_model)
            self.models[type_name] = model
            return model

        properties = schema.get("properties", {})
        fields = {}

        for field_name, field_def in properties.items():
            # Now returns (type, FieldInfo)
            fields[field_name] = self._json_type_to_python(field_def)

        model = create_model(type_name, **fields)
        self.models[type_name] = model
        return model

    def parse_all(self) -> Dict[str, Type[BaseModel]]:
        for type_name in self.raw_schemas:
            self.parse_type(type_name)
        return self.models
