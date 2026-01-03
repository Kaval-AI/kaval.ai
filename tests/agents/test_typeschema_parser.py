import pytest
from pydantic import BaseModel, ValidationError

from kavalai.agents.typeschema_parser import TypeSchemaParser


@pytest.fixture
def parser():
    return TypeSchemaParser()


def test_basic_types(parser):
    """Verify string, integer, and boolean mapping."""
    schema = {
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "is_active": {"type": "boolean"},
        },
        "required": ["name"],
    }

    Model = parser.parse_schema(schema, "User")
    user = Model(name="Alice", age=25, is_active=True)

    assert user.name == "Alice"
    assert user.age == 25
    assert user.is_active is True
    assert Model.model_fields["age"].is_required() is False


def test_nested_object_recursion(parser):
    """Verify that nested objects are converted into Pydantic models."""
    schema = {
        "properties": {
            "metadata": {
                "type": "object",
                "properties": {"version": {"type": "number"}},
            }
        }
    }

    Model = parser.parse_schema(schema, "Project")
    data = {"metadata": {"version": 1.2}}
    project = Model(**data)

    assert project.metadata.version == 1.2
    assert isinstance(project.metadata, BaseModel)


def test_array_of_objects(parser):
    """Verify arrays containing complex objects."""
    schema = {
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "object", "properties": {"sku": {"type": "string"}}},
            }
        }
    }

    Model = parser.parse_schema(schema, "Cart")
    cart = Model(items=[{"sku": "A1"}, {"sku": "B2"}])

    assert len(cart.items) == 2
    assert cart.items[0].sku == "A1"
    assert isinstance(cart.items[0], BaseModel)


@pytest.mark.parametrize(
    "invalid_data",
    [
        {"age": "not-an-int"},  # Wrong type
        {},  # Missing required field 'name'
    ],
)
def test_validation_enforcement(parser, invalid_data):
    """Ensure the generated model strictly enforces the TypeSchema."""
    schema = {
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name"],
    }
    Model = parser.parse_schema(schema, "StrictModel")

    with pytest.raises(ValidationError):
        Model(**invalid_data)


def test_description_mapping(parser):
    """Verify that TypeSchema descriptions are passed to Pydantic Fields."""
    schema = {
        "properties": {"score": {"type": "number", "description": "The gaming score"}}
    }
    Model = parser.parse_schema(schema, "InfoModel")
    assert Model.model_fields["score"].description == "The gaming score"
