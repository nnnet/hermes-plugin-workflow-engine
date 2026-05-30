"""Sqlite-backed registry of workflow invocations.

Replaces the prior file-based ``<state_dir>/<workflow>/<session>.json``
layout with a transactional store inside the existing Hermes ``state.db``.

Why
---
- O(1) indexed lookup ("is there an active invocation for agent X /
  conversation Y / workflow Z?") instead of scanning a directory.
- Atomic transactions — concurrent turns can't corrupt each other.
- Native cancel / abandon / finish lifecycle — no orphan files.
- Multi-agent isolation by row, not by path collision.
- Observability — ``list_active()`` + janitor counts + retention.

What lives where
----------------
- **Active state**: ``state_blob`` TEXT column on each row. JSON-serialized
  ``WorkflowState`` payload.
- **Final artifacts**: still on disk at
  ``<state_dir>/<workflow>/artifacts/<invocation_id>.yaml`` — downstream
  skill consumers read them as files. Registry just links via id.
- **Per-turn audit snapshots (P7 history)**: still on disk under
  ``<state_dir>/<workflow>/history/<invocation_id>/v{NNNN}.json``. Same
  reason — replay tools read files.
- **Finished rows**: stay in the DB with ``finished_ts`` + ``finished_reason``
  set, until the janitor's retention sweep prunes them
  (``WORKFLOW_RETENTION_DAYS``, default 30).

Identity
--------
A unique skill invocation is identified by:
- ``invocation_id``: ``YYYYMMDDTHHMMSS-<rand4>`` UTC — primary key
- ``workflow_name``: e.g. ``desire-to-goal``
- ``agent_id``: ``main`` | ``chief-XYZ`` | ``worker-N`` — who triggered
- ``conversation_id``: stable per chat (e.g. ``tg-dm-548987``)

The plugin owns identity resolution: at the start of each turn it
either finds an active invocation matching (workflow, agent, conv) or
creates a new one. The ``invocation_id`` is then passed to the engine
verbatim until the workflow finishes.
"""
from __future__ import annotations

import json
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional

# Default to Hermes' own state.db. Overridable so tests can isolate
# (WORKFLOW_REGISTRY_DB=/tmp/test.db etc.).
DB_PATH_DEFAULT = os.environ.get(
    "WORKFLOW_REGISTRY_DB",
    "/opt/data/state.db",
)

# Phases that count as "still in flight" for lifecycle queries.
TERMINAL_REASONS = frozenset({"DONE", "FAILED", "CANCELLED", "ABANDONED_TIMEOUT"})

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS workflow_invocations (
    invocation_id     TEXT PRIMARY KEY,
    workflow_name     TEXT NOT NULL,
    agent_id          TEXT NOT NULL,
    conversation_id   TEXT NOT NULL,
    state_blob        TEXT NOT NULL DEFAULT '{}',
    phase             TEXT NOT NULL DEFAULT 'INIT',
    iteration         INTEGER NOT NULL DEFAULT 0,
    started_ts        REAL NOT NULL,
    last_active_ts    REAL NOT NULL,
    finished_ts       REAL,
    finished_reason   TEXT,
    is_test           INTEGER NOT NULL DEFAULT 0,
    parent_invocation TEXT
);
CREATE INDEX IF NOT EXISTS idx_active_invocations
ON workflow_invocations(workflow_name, agent_id, conversation_id, finished_ts);
CREATE INDEX IF NOT EXISTS idx_last_active
ON workflow_invocations(last_active_ts) WHERE finished_ts IS NULL;
CREATE INDEX IF NOT EXISTS idx_finished
ON workflow_invocations(finished_ts);
CREATE INDEX IF NOT EXISTS idx_is_test
ON workflow_invocations(is_test);
"""


@dataclass
class InvocationRecord:
    """One row of ``workflow_invocations`` mapped to a Python object."""

    invocation_id: str
    workflow_name: str
    agent_id: str
    conversation_id: str
    state_blob: str
    phase: str
    iteration: int
    started_ts: float
    last_active_ts: float
    finished_ts: Optional[float] = None
    finished_reason: Optional[str] = None
    is_test: bool = False
    parent_invocation: Optional[str] = None

    @classmethod
    def from_row(cls, row: tuple) -> "InvocationRecord":
        return cls(
            invocation_id=row[0],
            workflow_name=row[1],
            agent_id=row[2],
            conversation_id=row[3],
            state_blob=row[4],
            phase=row[5],
            iteration=row[6],
            started_ts=row[7],
            last_active_ts=row[8],
            finished_ts=row[9],
            finished_reason=row[10],
            is_test=bool(row[11]),
            parent_invocation=row[12],
        )

    @property
    def is_active(self) -> bool:
        return self.finished_ts is None

    def state_dict(self) -> dict[str, Any]:
        """Deserialize ``state_blob`` JSON to dict. ``{}`` on parse error."""
        if not self.state_blob:
            return {}
        try:
            return json.loads(self.state_blob)
        except (ValueError, TypeError):
            return {}


def _gen_invocation_id() -> str:
    """Identity format: ``YYYYMMDDTHHMMSS-<rand4hex>`` (UTC).

    Sortable, human-readable, collision-resistant within the same second
    (4-byte random suffix → 65k unique IDs per timestamp).
    """
    return (
        f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}"
        f"-{secrets.token_hex(2)}"
    )


class Registry:
    """CRUD for workflow invocations stored in sqlite.

    All methods open + close the connection per-call (sqlite is cheap
    for this and we want fresh schema/transaction state across uses
    from different processes — engine CLI subprocess, plugin in-process,
    janitor cron etc.).
    """

    def __init__(self, db_path: str = DB_PATH_DEFAULT):
        self.db_path = db_path
        self._ensure_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        # Default isolation_level (deferred). Commit explicitly per op.
        c = sqlite3.connect(self.db_path)
        try:
            yield c
        finally:
            c.close()

    def _ensure_schema(self) -> None:
        with self._conn() as c:
            c.executescript(SCHEMA_SQL)

    # ─── lookup ─────────────────────────────────────────────────────

    def find_active(
        self,
        workflow_name: str,
        agent_id: str,
        conversation_id: str,
    ) -> Optional[InvocationRecord]:
        """Return the most recent active invocation matching the triple."""
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM workflow_invocations "
                "WHERE workflow_name=? AND agent_id=? AND conversation_id=? "
                "AND finished_ts IS NULL "
                "ORDER BY last_active_ts DESC LIMIT 1",
                (workflow_name, agent_id, conversation_id),
            ).fetchone()
        return InvocationRecord.from_row(row) if row else None

    def get(self, invocation_id: str) -> Optional[InvocationRecord]:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM workflow_invocations WHERE invocation_id=?",
                (invocation_id,),
            ).fetchone()
        return InvocationRecord.from_row(row) if row else None

    # ─── mutation ───────────────────────────────────────────────────

    def start(
        self,
        workflow_name: str,
        agent_id: str,
        conversation_id: str,
        *,
        initial_state_blob: str = "{}",
        initial_phase: str = "INIT",
        is_test: bool = False,
        parent_invocation: Optional[str] = None,
        invocation_id: Optional[str] = None,
    ) -> InvocationRecord:
        """Create a fresh invocation. Returns the persisted record.

        Pass ``invocation_id`` to override the generated id (deterministic
        tests). Otherwise an UTC-timestamp + rand suffix is assigned.
        """
        invocation_id = invocation_id or _gen_invocation_id()
        now = time.time()
        with self._conn() as c:
            c.execute(
                "INSERT INTO workflow_invocations "
                "(invocation_id, workflow_name, agent_id, conversation_id, "
                " state_blob, phase, iteration, started_ts, last_active_ts, "
                " is_test, parent_invocation) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    invocation_id, workflow_name, agent_id, conversation_id,
                    initial_state_blob, initial_phase, 0, now, now,
                    1 if is_test else 0, parent_invocation,
                ),
            )
            c.commit()
        rec = self.get(invocation_id)
        assert rec is not None, "INSERT just-saved row not retrievable"
        return rec

    def save_state(
        self,
        invocation_id: str,
        state_blob: str,
        phase: str,
        iteration: int,
    ) -> None:
        """Persist a state mutation. Updates last_active_ts."""
        now = time.time()
        with self._conn() as c:
            c.execute(
                "UPDATE workflow_invocations "
                "SET state_blob=?, phase=?, iteration=?, last_active_ts=? "
                "WHERE invocation_id=? AND finished_ts IS NULL",
                (state_blob, phase, iteration, now, invocation_id),
            )
            c.commit()

    def finish(
        self,
        invocation_id: str,
        reason: str = "DONE",
    ) -> Optional[InvocationRecord]:
        """Mark an invocation as terminal. Idempotent on already-finished."""
        if reason not in TERMINAL_REASONS:
            # Accept arbitrary FAILED:<detail> for failure cases.
            if not reason.startswith("FAILED"):
                raise ValueError(
                    f"finish() reason must be in {TERMINAL_REASONS} "
                    f"or 'FAILED:<msg>'; got {reason!r}"
                )
        now = time.time()
        with self._conn() as c:
            c.execute(
                "UPDATE workflow_invocations "
                "SET finished_ts=?, finished_reason=? "
                "WHERE invocation_id=? AND finished_ts IS NULL",
                (now, reason, invocation_id),
            )
            c.commit()
        return self.get(invocation_id)

    def cancel(
        self,
        invocation_id: str,
        reason_detail: str = "user_cancel",
    ) -> Optional[InvocationRecord]:
        """Convenience wrapper for ``finish(reason='CANCELLED')``."""
        return self.finish(invocation_id, "CANCELLED")

    def cancel_active(
        self,
        workflow_name: str,
        agent_id: str,
        conversation_id: str,
        reason_detail: str = "user_cancel",
    ) -> Optional[InvocationRecord]:
        """Cancel whatever active invocation matches the triple, if any."""
        active = self.find_active(workflow_name, agent_id, conversation_id)
        if active is None:
            return None
        return self.cancel(active.invocation_id, reason_detail)

    # ─── queries ────────────────────────────────────────────────────

    def list_active(
        self,
        *,
        workflow_name: Optional[str] = None,
        agent_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        is_test: Optional[bool] = None,
    ) -> list[InvocationRecord]:
        clauses = ["finished_ts IS NULL"]
        params: list[Any] = []
        for col, val in (
            ("workflow_name", workflow_name),
            ("agent_id", agent_id),
            ("conversation_id", conversation_id),
        ):
            if val is not None:
                clauses.append(f"{col} = ?")
                params.append(val)
        if is_test is not None:
            clauses.append("is_test = ?")
            params.append(1 if is_test else 0)
        with self._conn() as c:
            rows = c.execute(
                f"SELECT * FROM workflow_invocations "
                f"WHERE {' AND '.join(clauses)} "
                f"ORDER BY last_active_ts DESC",
                params,
            ).fetchall()
        return [InvocationRecord.from_row(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        """Quick observability snapshot — for dashboards / debug CLI."""
        with self._conn() as c:
            total = c.execute(
                "SELECT count(*) FROM workflow_invocations"
            ).fetchone()[0]
            active = c.execute(
                "SELECT count(*) FROM workflow_invocations "
                "WHERE finished_ts IS NULL"
            ).fetchone()[0]
            by_reason = dict(c.execute(
                "SELECT COALESCE(finished_reason, 'IN_FLIGHT'), count(*) "
                "FROM workflow_invocations GROUP BY 1"
            ).fetchall())
            by_workflow = dict(c.execute(
                "SELECT workflow_name, count(*) "
                "FROM workflow_invocations WHERE finished_ts IS NULL "
                "GROUP BY workflow_name"
            ).fetchall())
        return {
            "total_rows": total,
            "active": active,
            "by_reason": by_reason,
            "active_by_workflow": by_workflow,
        }

    # ─── janitor / cleanup ──────────────────────────────────────────

    def janitor_sweep(
        self,
        *,
        default_timeout_sec: float = 10800,  # 3h
        test_timeout_sec: float = 300,        # 5min for is_test=1
        now: Optional[float] = None,
    ) -> list[InvocationRecord]:
        """Mark stale active invocations as ABANDONED_TIMEOUT.

        Test invocations get a much tighter timeout so F1 / smoke runs
        don't leave debris. Returns the affected rows (POST-update — so
        their ``finished_ts`` is set).
        """
        now = now if now is not None else time.time()
        cutoff_default = now - default_timeout_sec
        cutoff_test = now - test_timeout_sec
        with self._conn() as c:
            stale_ids = [
                row[0] for row in c.execute(
                    "SELECT invocation_id FROM workflow_invocations "
                    "WHERE finished_ts IS NULL AND ("
                    "  (is_test = 1 AND last_active_ts < ?) "
                    "  OR (is_test = 0 AND last_active_ts < ?)"
                    ")",
                    (cutoff_test, cutoff_default),
                ).fetchall()
            ]
            for inv_id in stale_ids:
                c.execute(
                    "UPDATE workflow_invocations "
                    "SET finished_ts=?, finished_reason='ABANDONED_TIMEOUT' "
                    "WHERE invocation_id=?",
                    (now, inv_id),
                )
            c.commit()
        return [self.get(i) for i in stale_ids if self.get(i) is not None]  # type: ignore[misc]

    def delete_test_invocations(self) -> int:
        """Hard-delete every is_test=1 row. F1 isolation primitive."""
        with self._conn() as c:
            cur = c.execute(
                "DELETE FROM workflow_invocations WHERE is_test = 1"
            )
            c.commit()
            return cur.rowcount

    def delete_by_filter(
        self,
        *,
        workflow_name: Optional[str] = None,
        agent_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        is_test: Optional[bool] = None,
        only_finished: bool = False,
    ) -> int:
        """Hard-delete rows matching filter. ``only_finished=True``
        protects active invocations from accidental nuke."""
        clauses: list[str] = []
        params: list[Any] = []
        for col, val in (
            ("workflow_name", workflow_name),
            ("agent_id", agent_id),
            ("conversation_id", conversation_id),
        ):
            if val is not None:
                clauses.append(f"{col} = ?")
                params.append(val)
        if is_test is not None:
            clauses.append("is_test = ?")
            params.append(1 if is_test else 0)
        if only_finished:
            clauses.append("finished_ts IS NOT NULL")
        where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._conn() as c:
            cur = c.execute(
                f"DELETE FROM workflow_invocations{where_sql}", params,
            )
            c.commit()
            return cur.rowcount

    def purge_finished_older_than(self, *, days: int = 30) -> int:
        """Retention: hard-delete finished rows older than ``days``."""
        cutoff = time.time() - days * 86400
        with self._conn() as c:
            cur = c.execute(
                "DELETE FROM workflow_invocations "
                "WHERE finished_ts IS NOT NULL AND finished_ts < ?",
                (cutoff,),
            )
            c.commit()
            return cur.rowcount
