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
        self.auth = (username, password) if username and password else None
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

    async def stream_agent(self, data: BaseModel, external_id: Optional[str] = None):
        if self.input_schema is None or self.output_schema is None:
            await self.discover_schemas()

        payload = {
            "session_id": self.session_id,
            "external_id": external_id,
            "data": data.model_dump(),
        }

        url = f"{self.base_url}/stream_agent"

        async with httpx.AsyncClient(auth=self.auth, timeout=self.timeout) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        yield line[6:]
