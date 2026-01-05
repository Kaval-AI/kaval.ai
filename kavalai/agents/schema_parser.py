from typing import Any, Dict, Type

from pydantic import BaseModel, create_model


class SchemaParser:
    def __init__(self, datatypes: Dict[str, Any]):
        self.raw_schemas = datatypes
        self.models: Dict[str, Type[BaseModel]] = {}

    def _json_type_to_python(self, prop_def: Dict[str, Any]) -> Any:
        """Maps JSON schema types to Python types."""
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
            return self.parse_type(ref_name)

        if prop_def.get("type") == "array":
            items_def = prop_def.get("items", {})
            item_type = self._json_type_to_python(items_def)
            return list[item_type]

        json_type = prop_def.get("type", "string")
        return type_map.get(json_type, Any)

    def parse_type(self, type_name: str) -> Type[BaseModel]:
        """Recursively parses a schema definition into a Pydantic model."""
        if type_name in self.models:
            return self.models[type_name]

        schema = self.raw_schemas.get(type_name)
        if not schema:
            raise ValueError(f"Definition for '{type_name}' not found.")

        properties = schema.get("properties", {})
        fields = {}

        for field_name, field_def in properties.items():
            # Determine the type of the field
            field_type = self._json_type_to_python(field_def)
            # In this simple parser, we treat all as optional or default to None
            # You can expand this to check for 'required' arrays in the schema
            fields[field_name] = (field_type, ...)

        # Dynamically create the Pydantic model
        model = create_model(type_name, **fields)
        self.models[type_name] = model
        return model

    def parse_all(self) -> Dict[str, Type[BaseModel]]:
        """Parses all keys in the datatypes dict."""
        for type_name in self.raw_schemas:
            self.parse_type(type_name)
        return self.models
