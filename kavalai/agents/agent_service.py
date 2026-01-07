import uuid
from copy import copy
from typing import Optional
from uuid import UUID


class AgentServiceException(Exception):
    pass


class AgentService:
    """Simple agent service that manages interaction history."""

    def __init__(self):
        self.__agents = {}
        self.__interactions = {}

    def get_agent_id(self, agent_name: str) -> Optional[str]:
        if agent_name not in self.__agents:
            raise KeyError(f"Agent <{agent_name}> does not exist!")
        return self.__agents.get(agent_name)

    def create_agent(self, agent_name: str) -> UUID:
        if agent_name not in self.__agents:
            new_uuid = uuid.uuid4()
            self.__agents[agent_name] = new_uuid
            return new_uuid
        raise AgentServiceException(f"Agent <{agent_name}> already exists!")

    def create_session(self, agent_name: str) -> UUID:
        if agent_name not in self.__agents:
            raise KeyError(f"Agent <{agent_name}> does not exist!")
        new_uuid = uuid.uuid4()
        self.__interactions[new_uuid] = []
        return new_uuid

    def add_message(self, interaction_id: UUID, role: str, message: str) -> None:
        if interaction_id not in self.__interactions:
            raise AgentServiceException(
                f"Interaction <{interaction_id}> does not exist!"
            )
        if role not in ("assistant", "user"):
            raise Exception(
                "Only can add user and assistant messages to interaction history."
            )
        self.__interactions[interaction_id].append((role, message))

    def get_message_history(self, interaction_id: UUID) -> list[str]:
        if interaction_id not in self.__interactions:
            raise AgentServiceException(
                f"Interaction <{interaction_id}> does not exist!"
            )
        return copy(self.__interactions[interaction_id])
