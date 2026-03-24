from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def state_directory() -> Path:
    return Path("~/.local/state/murmur").expanduser()


def service_state_path() -> Path:
    return state_directory() / "service.json"


def service_log_path() -> Path:
    return state_directory() / "service.log"


def transcript_db_path() -> Path:
    return state_directory() / "transcripts.sqlite3"


def ensure_state_directory() -> Path:
    path = state_directory()
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class ServiceState:
    pid: int
    host: str
    port: int
    started_at: str
    status_indicator_pid: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "host": self.host,
            "port": self.port,
            "started_at": self.started_at,
            "status_indicator_pid": self.status_indicator_pid,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ServiceState:
        return cls(
            pid=int(payload["pid"]),
            host=str(payload["host"]),
            port=int(payload["port"]),
            started_at=str(payload.get("started_at") or ""),
            status_indicator_pid=(
                int(payload["status_indicator_pid"])
                if payload.get("status_indicator_pid") is not None
                else None
            ),
        )

    @classmethod
    def new(
        cls,
        *,
        pid: int,
        host: str,
        port: int,
        status_indicator_pid: int | None = None,
    ) -> ServiceState:
        return cls(
            pid=pid,
            host=host,
            port=port,
            started_at=datetime.now(timezone.utc).isoformat(),
            status_indicator_pid=status_indicator_pid,
        )


@dataclass(frozen=True)
class ServiceStatus:
    running: bool
    pid: int | None
    host: str | None
    port: int | None
    started_at: str | None
    status_indicator_pid: int | None
    stale: bool
    reachable: bool
    state_path: Path
    pid_alive: bool = False
    stale_reason: str | None = None
