"""Transactional synthetic tools and durable operator queue. No payment execution."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from uuid import UUID, uuid4

from supportpilot.common.schemas import LanguageCode
from supportpilot.service.contracts import Escalation, ServiceError


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS orders (
                    id TEXT PRIMARY KEY, customer TEXT NOT NULL, status TEXT NOT NULL,
                    payment TEXT NOT NULL, cancellation_requested INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS stock (sku TEXT PRIMARY KEY, quantity INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS escalations (
                    id TEXT PRIMARY KEY, customer TEXT NOT NULL, session TEXT NOT NULL,
                    status TEXT NOT NULL, reason TEXT NOT NULL, language TEXT NOT NULL,
                    context TEXT NOT NULL, evidence TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS receipts (
                    customer TEXT NOT NULL, request TEXT NOT NULL, fingerprint TEXT NOT NULL,
                    result TEXT NOT NULL, PRIMARY KEY(customer, request));
            """)

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=1)
        db.row_factory = sqlite3.Row
        return db

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        db = self.connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def seed_demo(self) -> None:
        """Explicit demo initialization; existing mutable state is never overwritten."""
        with self.transaction() as db:
            db.executemany(
                "INSERT OR IGNORE INTO orders(id,customer,status,payment) VALUES(?,?,?,?)",
                [
                    ("SP-1001", "demo_alice", "processing", "paid"),
                    ("SP-1002", "demo_bob", "shipped", "paid"),
                ],
            )
            db.execute("INSERT OR IGNORE INTO stock VALUES('TV-5501',7)")

    def tool(
        self, db: sqlite3.Connection, customer: str, name: str, reference: str
    ) -> dict[str, str | int]:
        if name == "get_stock":
            row = db.execute("SELECT sku,quantity FROM stock WHERE sku=?", (reference,)).fetchone()
            if row is None:
                raise ServiceError(404, "reference_unavailable")
            return {"sku": row["sku"], "quantity": row["quantity"]}
        if name not in {"get_order", "get_payment", "request_cancellation"}:
            raise ServiceError(400, "unsupported_operation")
        row = db.execute(
            "SELECT * FROM orders WHERE id=? AND customer=?", (reference, customer)
        ).fetchone()
        # Absent and someone else's order have exactly the same observable failure.
        if row is None:
            raise ServiceError(404, "reference_unavailable")
        if name == "get_payment":
            return {"order_id": row["id"], "payment_status": row["payment"]}
        if name == "request_cancellation":
            if row["status"] != "processing":
                raise ServiceError(409, "operation_unavailable")
            db.execute(
                "UPDATE orders SET cancellation_requested=1 WHERE id=? AND customer=?",
                (reference, customer),
            )
            return {"order_id": row["id"], "cancellation_request": "recorded"}
        return {"order_id": row["id"], "order_status": row["status"]}

    def enqueue(
        self,
        db: sqlite3.Connection,
        customer: str,
        session: UUID,
        reason: str,
        language: LanguageCode,
        context: list[str],
        evidence: list[str],
    ) -> UUID:
        row = db.execute(
            "SELECT id,reason FROM escalations "
            "WHERE customer=? AND session=? AND status!='resolved'",
            (customer, str(session)),
        ).fetchone()
        if row:
            if row["reason"] in {
                "independent_safety_guard",
                "safety_risk",
                "legal_threat",
                "privacy_request",
            }:
                reason = row["reason"]
            db.execute(
                "UPDATE escalations SET reason=?,context=?,evidence=? WHERE id=?",
                (reason, json.dumps(context), json.dumps(evidence), row["id"]),
            )
            return UUID(row["id"])
        identity = uuid4()
        db.execute(
            "INSERT INTO escalations VALUES(?,?,?,'open',?,?,?,?,?)",
            (
                str(identity),
                customer,
                str(session),
                reason,
                language,
                json.dumps(context),
                json.dumps(evidence),
                time.time(),
            ),
        )
        return identity

    def escalations(self, customer: str) -> list[Escalation]:
        with closing(self.connect()) as db:
            if customer == "operator":
                rows = db.execute("SELECT * FROM escalations ORDER BY created DESC LIMIT 100")
            else:
                rows = db.execute(
                    "SELECT * FROM escalations WHERE customer=? ORDER BY created DESC LIMIT 100",
                    (customer,),
                )
            return [
                Escalation(
                    escalation_id=r["id"],
                    session_id=r["session"],
                    status=r["status"],
                    reason=r["reason"],
                    language=r["language"],
                    context=json.loads(r["context"]),
                    evidence=json.loads(r["evidence"]),
                    created_at=r["created"],
                )
                for r in rows
            ]

    def transition(self, identity: UUID, status: str) -> None:
        with self.transaction() as db:
            previous = "open" if status == "in_progress" else "in_progress"
            if (
                db.execute(
                    "UPDATE escalations SET status=? WHERE id=? AND status=?",
                    (status, str(identity), previous),
                ).rowcount
                != 1
            ):
                raise ServiceError(409, "invalid_transition")
