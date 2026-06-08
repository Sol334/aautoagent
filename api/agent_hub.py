import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from api.schemas import AgentCard, JsonRpcRequest, JsonRpcResponse, Message, Part, Task

log = logging.getLogger(__name__)

_AGENTS: dict[str, Any] = {}


# ── TaskStore ─────────────────────────────────────────────────────────────────

class TaskStore:
    """Redis-backed task store with automatic in-memory fallback."""

    _TTL = 3600  # 1 hour

    def __init__(self, redis_url: str = "redis://localhost:6380"):
        self._redis = None
        self._memory: dict[str, dict] = {}
        try:
            import redis
            r = redis.from_url(redis_url, socket_connect_timeout=1, socket_timeout=1)
            r.ping()
            self._redis = r
            log.info("TaskStore: connected to Redis at %s", redis_url)
        except Exception as exc:
            log.info("TaskStore: Redis unavailable (%s) — using in-memory fallback", exc)

    def set(self, task_id: str, task: dict) -> None:
        payload = json.dumps(task)
        if self._redis:
            try:
                self._redis.setex(task_id, self._TTL, payload)
                return
            except Exception:
                pass
        self._memory[task_id] = {"payload": payload, "expires": time.time() + self._TTL}
        # Evict expired in-memory entries (keep it small)
        now = time.time()
        self._memory = {k: v for k, v in self._memory.items() if v["expires"] > now}

    def get(self, task_id: str) -> dict | None:
        if self._redis:
            try:
                raw = self._redis.get(task_id)
                return json.loads(raw) if raw else None
            except Exception:
                pass
        entry = self._memory.get(task_id)
        if entry and entry["expires"] > time.time():
            return json.loads(entry["payload"])
        return None

    @property
    def backend(self) -> str:
        return "redis" if self._redis else "memory"


_task_store = TaskStore(os.getenv("REDIS_URL", "redis://localhost:6380"))


@asynccontextmanager
async def _lifespan(app: FastAPI):
    from api.agents.watchdog_agent import WatchdogAgent
    from api.agents.analyst_agent import AnalystAgent
    from api.agents.trader_agent import TraderAgent

    for cls in (WatchdogAgent, AnalystAgent, TraderAgent):
        inst = cls()
        _AGENTS[inst.name] = inst
    log.info("Registered agents: %s", list(_AGENTS.keys()))
    yield
    _AGENTS.clear()


app = FastAPI(title="Galactic Capital — Agent Hub", version="1.0.0", lifespan=_lifespan)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "agents": list(_AGENTS.keys()),
        "task_store": _task_store.backend,
    }


@app.get("/agents")
def list_agents() -> list[AgentCard]:
    return [
        AgentCard(name=a.name, description=a.description, skills=a.skills)
        for a in _AGENTS.values()
    ]


def _dispatch(agent_name: str, request: JsonRpcRequest) -> JSONResponse:
    """Core JSON-RPC 2.0 dispatcher shared by both routing endpoints."""
    if request.method != "tasks/send":
        return JSONResponse(
            JsonRpcResponse(
                id=request.id,
                error={"code": -32601, "message": f"Method not found: {request.method}"},
            ).model_dump()
        )

    try:
        message_raw = request.params.get("message", {})
        parts = message_raw.get("parts", [])
        text = parts[0].get("text", "") if parts else ""
    except Exception as exc:
        return JSONResponse(
            JsonRpcResponse(
                id=request.id,
                error={"code": -32602, "message": f"Invalid params: {exc}"},
            ).model_dump()
        )

    agent = _AGENTS.get(agent_name)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    task_id = str(uuid.uuid4())
    try:
        result_text = agent.handle(text)
        task = Task(
            id=task_id,
            status="completed",
            message=Message(role="user", parts=[Part(text=text)]),
            result=result_text,
        )
        task_dict = task.model_dump()
        _task_store.set(task_id, task_dict)
        return JSONResponse(
            JsonRpcResponse(id=request.id, result=task_dict).model_dump()
        )
    except Exception as exc:
        log.exception("Agent '%s' raised an unhandled exception", agent_name)
        return JSONResponse(
            JsonRpcResponse(
                id=request.id,
                error={"code": -32603, "message": str(exc)},
            ).model_dump()
        )


@app.post("/agents/{agent_name}")
async def call_agent(agent_name: str, request: JsonRpcRequest) -> JSONResponse:
    if agent_name not in _AGENTS:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    return _dispatch(agent_name, request)


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    task = _task_store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found or expired")
    return {"jsonrpc": "2.0", "result": task}


@app.post("/tasks/send")
async def auto_route(request: JsonRpcRequest) -> JSONResponse:
    """Auto-route by scanning message text against each agent's name and skills."""
    try:
        message_raw = request.params.get("message", {})
        parts = message_raw.get("parts", [])
        text = parts[0].get("text", "") if parts else ""
    except Exception:
        text = ""

    text_lower = text.lower()

    matched_agent: str | None = None
    for name, agent in _AGENTS.items():
        if name in text_lower:
            matched_agent = name
            break
        for skill in agent.skills:
            if skill.replace("_", " ") in text_lower or skill in text_lower:
                matched_agent = name
                break
        if matched_agent:
            break

    # Fall back to watchdog when nothing matches
    if matched_agent is None:
        matched_agent = "watchdog"

    return _dispatch(matched_agent, request)
