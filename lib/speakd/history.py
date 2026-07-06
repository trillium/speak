"""SQLite-backed history of spoken text."""

import os
import sqlite3

from .text import split_clauses


class SpeechHistory:
    """Persistent history of spoken text entries."""

    def __init__(self):
        db_path = f"/tmp/speak-{os.environ['USER']}-history.db"
        self._db = sqlite3.connect(db_path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS history ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  text TEXT NOT NULL,"
            "  spoken_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))"
            ")"
        )
        self._db.commit()
        self._migrate()

    def _migrate(self):
        """Add columns if they don't exist yet."""
        cols = {row[1] for row in self._db.execute("PRAGMA table_info(history)")}
        if "caller" not in cols:
            self._db.execute("ALTER TABLE history ADD COLUMN caller TEXT NOT NULL DEFAULT ''")
        if "session" not in cols:
            self._db.execute("ALTER TABLE history ADD COLUMN session TEXT NOT NULL DEFAULT ''")
        if "voice" not in cols:
            self._db.execute("ALTER TABLE history ADD COLUMN voice TEXT NOT NULL DEFAULT ''")
        self._db.commit()

    def record(self, text: str, caller: str = "", session: str = "",
               voice: str = "") -> int:
        """Insert a history row and return its ID."""
        cur = self._db.execute(
            "INSERT INTO history (text, caller, session, voice) VALUES (?, ?, ?, ?)",
            (text, caller, session, voice),
        )
        self._db.commit()
        return cur.lastrowid

    def update_voice(self, row_id: int, voice: str) -> None:
        """Update the voice for a history row (set after pool resolution)."""
        self._db.execute(
            "UPDATE history SET voice = ? WHERE id = ?", (voice, row_id)
        )
        self._db.commit()

    @staticmethod
    def _row_to_entry(r) -> dict:
        text = r[1]
        return {
            "id": r[0], "text": text, "caller": r[2],
            "session": r[3], "voice": r[4], "spoken_at": r[5],
            "clauses": split_clauses(text),
        }

    def get_by_id(self, row_id: int) -> dict | None:
        """Look up a history entry by ID. Returns dict or None."""
        row = self._db.execute(
            "SELECT id, text, caller, session, voice, spoken_at "
            "FROM history WHERE id = ?", (row_id,)
        ).fetchone()
        return self._row_to_entry(row) if row is not None else None

    def get(self, n: int = 10, offset: int = 0) -> tuple[list[dict], int]:
        total = self._db.execute(
            "SELECT COUNT(*) FROM history"
        ).fetchone()[0]
        rows = self._db.execute(
            "SELECT id, text, caller, session, voice, spoken_at "
            "FROM history ORDER BY id DESC LIMIT ? OFFSET ?",
            (n, offset),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows], total

    def get_by_session(self, session: str, n: int = 10, offset: int = 0) -> tuple[list[dict], int]:
        total = self._db.execute(
            "SELECT COUNT(*) FROM history WHERE session = ?", (session,)
        ).fetchone()[0]
        rows = self._db.execute(
            "SELECT id, text, caller, session, voice, spoken_at "
            "FROM history WHERE session = ? ORDER BY id DESC LIMIT ? OFFSET ?",
            (session, n, offset),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows], total

    def get_by_caller(self, caller: str, n: int = 10, offset: int = 0) -> tuple[list[dict], int]:
        total = self._db.execute(
            "SELECT COUNT(*) FROM history WHERE caller = ?", (caller,)
        ).fetchone()[0]
        rows = self._db.execute(
            "SELECT id, text, caller, session, voice, spoken_at "
            "FROM history WHERE caller = ? ORDER BY id DESC LIMIT ? OFFSET ?",
            (caller, n, offset),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows], total

    def last_voice_for_caller(self, caller: str) -> str | None:
        """Return the most recently used voice for a caller, or None."""
        row = self._db.execute(
            "SELECT voice FROM history WHERE caller = ? AND voice != '' "
            "ORDER BY id DESC LIMIT 1",
            (caller,),
        ).fetchone()
        return row[0] if row else None
