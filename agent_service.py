"""Live help-desk: agent presence, visitor sessions, message routing."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
AGENT_TIMEOUT = timedelta(seconds=90)
MAX_MESSAGES = 500


class AgentService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._agents: dict[str, dict] = {}
        self._sessions: dict[str, dict] = {}

    def _now(self) -> datetime:
        return datetime.now(IST)

    def _iso(self, dt: datetime) -> str:
        return dt.isoformat()

    def _prune_agents(self) -> None:
        cutoff = self._now() - AGENT_TIMEOUT
        for aid in [k for k, v in self._agents.items() if v["last_seen"] < cutoff]:
            del self._agents[aid]

    def has_online_agent(self) -> bool:
        with self._lock:
            self._prune_agents()
            return bool(self._agents)

    def agent_status(self) -> dict:
        with self._lock:
            self._prune_agents()
            agents = [{"id": k, "name": v["name"]} for k, v in self._agents.items()]
            return {"online": bool(agents), "count": len(agents), "agents": agents}

    def register_agent(self, name: str = "Agent") -> dict:
        agent_id = str(uuid.uuid4())
        now = self._now()
        name = (name or "Agent").strip() or "Agent"
        with self._lock:
            self._agents[agent_id] = {"name": name, "last_seen": now}
        return {"agent_id": agent_id, "name": name}

    def heartbeat(self, agent_id: str) -> bool:
        with self._lock:
            info = self._agents.get(agent_id)
            if not info:
                return False
            info["last_seen"] = self._now()
            return True

    def unregister(self, agent_id: str) -> bool:
        with self._lock:
            return self._agents.pop(agent_id, None) is not None

    def create_session(self) -> dict:
        sid = str(uuid.uuid4())
        now = self._now()
        session = {
            "id": sid,
            "label": f"User {sid[:8]}",
            "agent_id": None,
            "status": "open",
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }
        with self._lock:
            self._sessions[sid] = session
        return self._public(session)

    def _public(self, s: dict) -> dict:
        return {
            "id": s["id"],
            "label": s["label"],
            "agent_id": s["agent_id"],
            "status": s["status"],
            "created_at": self._iso(s["created_at"]),
            "updated_at": self._iso(s["updated_at"]),
            "messages": list(s["messages"]),
        }

    def get_session(self, session_id: str) -> dict | None:
        with self._lock:
            s = self._sessions.get(session_id)
            return self._public(s) if s else None

    def list_sessions(self, agent_id: str | None = None) -> list[dict]:
        with self._lock:
            self._prune_agents()
            out = []
            for s in self._sessions.values():
                if s["status"] != "open":
                    continue
                unread = sum(1 for m in s["messages"] if m["role"] == "visitor" and not m.get("read"))
                last = s["messages"][-1]["text"] if s["messages"] else ""
                out.append({
                    "id": s["id"],
                    "label": s["label"],
                    "unread": unread,
                    "last_message": last,
                    "updated_at": self._iso(s["updated_at"]),
                    "agent_id": s["agent_id"],
                })
            out.sort(key=lambda x: x["updated_at"], reverse=True)
            return out

    def claim(self, session_id: str, agent_id: str) -> dict | None:
        with self._lock:
            self._prune_agents()
            if agent_id not in self._agents:
                return None
            s = self._sessions.get(session_id)
            if not s or s["status"] != "open":
                return None
            if s["agent_id"] and s["agent_id"] != agent_id:
                return None
            if not s["agent_id"]:
                s["agent_id"] = agent_id
                now = self._now()
                s["updated_at"] = now
                s["messages"].append({
                    "id": str(uuid.uuid4()),
                    "role": "system",
                    "text": "An agent joined the chat.",
                    "at": self._iso(now),
                })
            return self._public(s)

    def add_message(self, session_id: str, role: str, text: str, *, agent_id: str | None = None) -> dict | None:
        text = (text or "").strip()
        if not text:
            return None
        now = self._now()
        msg = {"id": str(uuid.uuid4()), "role": role, "text": text, "at": self._iso(now)}
        with self._lock:
            s = self._sessions.get(session_id)
            if not s or s["status"] != "open":
                return None
            if role == "agent":
                if not agent_id or s["agent_id"] != agent_id:
                    return None
                msg["read"] = True
            elif role == "visitor":
                msg["read"] = False
            s["messages"].append(msg)
            if len(s["messages"]) > MAX_MESSAGES:
                s["messages"] = s["messages"][-MAX_MESSAGES:]
            s["updated_at"] = now
            return {"message": msg, "session": self._public(s)}

    def mark_read(self, session_id: str, agent_id: str) -> bool:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s or s["agent_id"] != agent_id:
                return False
            for m in s["messages"]:
                if m["role"] == "visitor":
                    m["read"] = True
            return True

    def visitor_poll(self, session_id: str, after: int = 0) -> tuple[list[dict], int]:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                return [], after
            msgs = s["messages"]
            new = [m for m in msgs[after:] if m["role"] in ("agent", "system")]
            return new, len(msgs)


_service: AgentService | None = None


def get_agent_service() -> AgentService:
    global _service
    if _service is None:
        _service = AgentService()
    return _service
