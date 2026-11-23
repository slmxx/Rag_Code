"""Destination : rag_pipeline/cache.py

Petit cache clé -> JSON sur disque (SQLite). Sert à ne jamais repayer un appel de génération ou de jugement :
la même question avec les mêmes extraits redonne la même réponse, sans nouvel appel.
"""
from __future__ import annotations
import threading
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from rag_pipeline import config


class JsonCache:
    def __init__(self, name: str):
        cache_dir = Path(getattr(config, "CACHE_DIR", config.PROJECT_ROOT / ".embedding_cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(cache_dir / f"{name}.sqlite3"), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self._db.commit()

    @staticmethod
    def key(*parts: Any) -> str:
        raw = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Any | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (key, json.dumps(value, ensure_ascii=False)))
            self._db.commit()