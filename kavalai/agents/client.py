import urllib.parse
from typing import Optional, Type

import httpx
from json_schema_to_pydantic import create_model
from pydantic import BaseModel

from kavalai.tools.openapi_spec_parser import OpenApiSpecParser


class AgentClient:
    def __init__(
        self,
        base_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.auth = (username, password) if username else None
        self.timeout = timeout
        self.session_id: Optional[str] = None
        self.input_schema: Optional[Type[BaseModel]] = None
        self.output_schema: Optional[Type[BaseModel]] = None

    async def discover_schemas(self):
        async with httpx.AsyncClient(auth=self.auth, timeout=self.timeout) as client:
            openapi_spec_url = urllib.parse.urljoin(self.base_url + "/", "openapi.json")
            resp = await client.get(openapi_spec_url)
            resp.raise_for_status()
            spec = resp.json()

        parser = OpenApiSpecParser(spec)

        self.input_schema = (
            create_model(parser.get_path_request_schema("/run_agent", "POST"))
            .model_fields["data"]
            .annotation
        )
        self.output_schema = (
            create_model(parser.get_path_response_schema("/run_agent", "POST"))
            .model_fields["data"]
            .annotation
        )

    async def run_agent(
        self, data: BaseModel, external_id: Optional[str] = None
    ) -> BaseModel:
        if self.input_schema is None or self.output_schema is None:
            await self.discover_schemas()

        payload = {
            "session_id": self.session_id,
            "external_id": external_id,
            "data": data.model_dump(),
        }

        url = f"{self.base_url}/run_agent"

        async with httpx.AsyncClient(auth=self.auth, timeout=self.timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            response_json = resp.json()

        self.session_id = response_json.get("session_id")
        return self.output_schema(**response_json["data"])
