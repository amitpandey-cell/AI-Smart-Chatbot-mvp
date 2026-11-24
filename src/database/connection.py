"""Database connection manager with MySQL pooling and automated SQLite fallback.

Enables seamless development and automated offline testing even when a live MySQL
instance is unavailable.
"""

from contextlib import contextmanager
import os
import sqlite3
from typing import Any, Dict, Generator, List, Optional, Tuple
import pymysql
from pymysql.cursors import DictCursor

from src.config import get_settings
from src.safety.logger import get_logger

logger = get_logger(__name__)


_mysql_failed_globally: bool = False


class DatabaseManager:
    """Manages database connectivity with MySQL priority and SQLite fallback."""

    def __init__(self, override_sqlite_path: Optional[str] = None):
        self.settings = get_settings()
        self.override_sqlite_path = override_sqlite_path
        self._db_type: Optional[str] = None
        self._init_connection()

    @property
    def db_type(self) -> str:
        """Active database backend: 'mysql' or 'sqlite'."""
        return self._db_type or "uninitialized"

    def _init_connection(self) -> None:
        """Attempt MySQL connection; fall back to SQLite upon failure."""
        global _mysql_failed_globally
        force_sqlite = (
            self.override_sqlite_path is not None
            or os.getenv("FORCE_SQLITE", "false").lower() in ("true", "1", "yes")
            or _mysql_failed_globally
        )

        if force_sqlite:
            self._db_type = "sqlite"
            logger.info("DatabaseManager initialized in SQLite mode: %s", self._get_sqlite_path())
            self._ensure_sqlite_schema()
            return

        # Attempt MySQL with quick connect timeout
        try:
            conn = pymysql.connect(
                host=self.settings.DB_HOST,
                port=self.settings.DB_PORT,
                user=self.settings.DB_USER,
                password=self.settings.DB_PASSWORD,
                database=self.settings.DB_NAME,
                connect_timeout=1,
                cursorclass=DictCursor,
            )
            conn.close()
            self._db_type = "mysql"
            logger.info("Successfully connected to MySQL database: %s", self.settings.DB_NAME)
        except Exception as exc:
            _mysql_failed_globally = True
            logger.warning(
                "MySQL connection to %s:%d failed (%s). Falling back to local SQLite: %s",
                self.settings.DB_HOST,
                self.settings.DB_PORT,
                exc,
                self._get_sqlite_path(),
            )
            self._db_type = "sqlite"
            self._ensure_sqlite_schema()

    def _get_sqlite_path(self) -> str:
        return self.override_sqlite_path or self.settings.SQLITE_PATH

    def _ensure_sqlite_schema(self) -> None:
        """Create tables in SQLite if they do not exist."""
        path = self._get_sqlite_path()
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        
        cursor.execute("PRAGMA foreign_keys = ON;")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                char_length INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT DEFAULT 'New Conversation',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                citations_json TEXT NULL,
                audio_present BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
        """)
        conn.commit()
        conn.close()

    @contextmanager
    def get_connection(self) -> Generator[Any, None, None]:
        """Context manager yielding an active database connection."""
        if self._db_type == "mysql":
            conn = pymysql.connect(
                host=self.settings.DB_HOST,
                port=self.settings.DB_PORT,
                user=self.settings.DB_USER,
                password=self.settings.DB_PASSWORD,
                database=self.settings.DB_NAME,
                cursorclass=DictCursor,
                autocommit=False,
            )
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        else:
            conn = sqlite3.connect(self._get_sqlite_path())
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON;")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def execute(self, query: str, params: Optional[Tuple[Any, ...]] = None) -> int:
        """Execute write query (INSERT, UPDATE, DELETE) and return affected rows."""
        params = params or ()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self._db_type == "sqlite":
                # Convert %s placeholders to ? for SQLite
                sqlite_query = query.replace("%s", "?")
                cursor.execute(sqlite_query, params)
            else:
                cursor.execute(query, params)
            return cursor.rowcount

    def fetch_all(self, query: str, params: Optional[Tuple[Any, ...]] = None) -> List[Dict[str, Any]]:
        """Execute SELECT query and return list of dictionaries."""
        params = params or ()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self._db_type == "sqlite":
                sqlite_query = query.replace("%s", "?")
                cursor.execute(sqlite_query, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
            else:
                cursor.execute(query, params)
                return cursor.fetchall()

    def fetch_one(self, query: str, params: Optional[Tuple[Any, ...]] = None) -> Optional[Dict[str, Any]]:
        """Execute SELECT query and return single row dictionary or None."""
        rows = self.fetch_all(query, params)
        return rows[0] if rows else None

    def check_health(self) -> Dict[str, Any]:
        """Check active connection health."""
        try:
            res = self.fetch_one("SELECT 1 AS alive")
            connected = res is not None
        except Exception as exc:
            logger.error("Health check database query failed: %s", exc)
            connected = False

        return {
            "connected": connected,
            "db_type": self._db_type,
        }


# Global singleton database manager instance
_db_manager_instance: Optional[DatabaseManager] = None


def get_db_manager(override_sqlite_path: Optional[str] = None) -> DatabaseManager:
    """Retrieve or initialize global DatabaseManager singleton."""
    global _db_manager_instance
    if _db_manager_instance is None or override_sqlite_path is not None:
        _db_manager_instance = DatabaseManager(override_sqlite_path=override_sqlite_path)
    return _db_manager_instance
