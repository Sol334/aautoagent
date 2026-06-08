from pydantic import BaseModel
from typing import Any


class Part(BaseModel):
    type: str = "text"
    text: str


class Message(BaseModel):
    role: str  # "user" or "agent"
    parts: list[Part]


class AgentCard(BaseModel):
    name: str
    description: str
    skills: list[str]


class Task(BaseModel):
    id: str
    status: str = "submitted"  # submitted | running | completed | failed
    message: Message | None = None
    result: str | None = None
    error: str | None = None


class JsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: int | str
    method: str
    params: dict[str, Any] = {}


class JsonRpcResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: int | str
    result: Any = None
    error: dict | None = None
