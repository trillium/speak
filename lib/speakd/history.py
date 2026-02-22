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

    def record(self, text: str) -> None:
        self._db.execute("INSERT INTO history (text) VALUES (?)", (text,))
        self._db.commit()

    def get(self, n: int = 10) -> list[str]:
        rows = self._db.execute(
            "SELECT text FROM history ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [r[0] for r in reversed(rows)]
