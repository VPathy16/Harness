"""Append-only evidence and explicit verified result storage."""
import json
import sqlite3
from pathlib import Path

from .domain import verify


class Workspace:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "evidence.sqlite3")
        self.db.execute("CREATE TABLE IF NOT EXISTS results (id INTEGER PRIMARY KEY, task_id TEXT, task TEXT, result TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS skills (task_key TEXT PRIMARY KEY, expression TEXT NOT NULL)")

    @staticmethod
    def key(task):
        return json.dumps([sorted(task.numbers), task.target])

    def record(self, task, result):
        self.db.execute("INSERT INTO results(task_id,task,result) VALUES (?,?,?)",
                        (task.id, json.dumps(task.public()), json.dumps(result)))
        expression = result.get("expression")
        if result.get("status") == "verified" and verify(task, expression)[0]:
            self.db.execute("INSERT OR REPLACE INTO skills VALUES (?,?)", (self.key(task), expression))
        self.db.commit()

    def recall(self, task):
        row = self.db.execute("SELECT expression FROM skills WHERE task_key=?", (self.key(task),)).fetchone()
        if row and verify(task, row[0])[0]:
            return row[0]
        return None

    def close(self):
        self.db.close()
