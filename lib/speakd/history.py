"""SQLite-backed history of spoken text."""

import os
import sqlite3


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
        """Add caller and session columns if they don't exist yet."""
        cols = {row[1] for row in self._db.execute("PRAGMA table_info(history)")}
        if "caller" not in cols:
            self._db.execute("ALTER TABLE history ADD COLUMN caller TEXT NOT NULL DEFAULT ''")
        if "session" not in cols:
            self._db.execute("ALTER TABLE history ADD COLUMN session TEXT NOT NULL DEFAULT ''")
        self._db.commit()

    def record(self, text: str, caller: str = "", session: str = "") -> None:
        self._db.execute(
            "INSERT INTO history (text, caller, session) VALUES (?, ?, ?)",
            (text, caller, session),
        )
        self._db.commit()

    def get(self, n: int = 10) -> list[str]:
        rows = self._db.execute(
            "SELECT text FROM history ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [r[0] for r in reversed(rows)]

    def get_by_session(self, session: str, n: int = 10) -> list[str]:
        rows = self._db.execute(
            "SELECT text FROM history WHERE session = ? ORDER BY id DESC LIMIT ?",
            (session, n),
        ).fetchall()
        return [r[0] for r in reversed(rows)]

    def get_by_caller(self, caller: str, n: int = 10) -> list[str]:
        rows = self._db.execute(
            "SELECT text FROM history WHERE caller = ? ORDER BY id DESC LIMIT ?",
            (caller, n),
        ).fetchall()
        return [r[0] for r in reversed(rows)]
