import string
from textwrap import dedent

from pydantic import BaseModel


async def format_prompt(prompt_template: str, **kwargs: dict[str, BaseModel]):
    formatter = string.Formatter()
    required_fields = {
        field_name
        for (literal_text, field_name, format_spec, conversion) in formatter.parse(
            prompt_template
        )
    }
    required_context: dict[str, str | None] = {}
    for k, v in kwargs.items():
        if k in required_fields:
            if isinstance(v, BaseModel):
                required_context[k] = v.model_dump_json()
            else:
                required_context[k] = str(v)
    for field in required_fields:
        if field not in required_context:
            required_context[field] = None
    return dedent(prompt_template.strip()).format(**required_context)
