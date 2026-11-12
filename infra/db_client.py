"""TLS-verified, bounded MySQL pools; eval is the default connection mode."""

from contextlib import contextmanager, suppress
import os
from pathlib import Path
import ssl
from threading import Lock
from typing import Literal

from dbutils.pooled_db import PooledDB
from dotenv import load_dotenv
import pymysql


ROOT = Path(__file__).resolve().parents[1]
Mode = Literal["admin", "eval"]
_pools: dict[str, PooledDB] = {}
_pool_lock = Lock()


def load_environment() -> None:
    """Load the repo's optional .env without overriding injected credentials."""
    load_dotenv(ROOT / ".env", override=False)


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Missing environment variable: {name}")
    return value


def connection_options(mode: Mode = "eval") -> dict:
    load_environment()
    if mode not in ("admin", "eval"):
        raise ValueError("MySQL mode must be admin or eval")
    context = ssl.create_default_context()
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    # PEM text works in containers; a laptop's CA path does not.
    if ca_pem := os.environ.get("MYSQL_SSL_CA_PEM"):
        context.load_verify_locations(cadata=ca_pem)
    elif ca_path := os.environ.get("MYSQL_SSL_CA"):
        path = Path(ca_path).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        context.load_verify_locations(cafile=str(path))
    port = int(required_env("MYSQL_PORT"))
    if not 1 <= port <= 65535:
        raise ValueError("MYSQL_PORT must be between 1 and 65535")
    prefix = f"MYSQL_{mode.upper()}"
    return {
        "host": required_env("MYSQL_HOST"),
        "port": port,
        "user": required_env(f"{prefix}_USER"),
        "password": required_env(f"{prefix}_PASS"),
        # Admin must connect before the sandbox exists.
        "database": "sandbox_db" if mode == "eval" else None,
        "ssl": context,
        "charset": "utf8mb4",
        "cursorclass": pymysql.cursors.DictCursor,
        "connect_timeout": 10,
        "read_timeout": 15,
        "write_timeout": 15,
        "autocommit": False,
        "local_infile": False,
    }


def get_pool(mode: Mode = "eval") -> PooledDB:
    """Return a lazy pool per mode in this process (at most four connections)."""
    with _pool_lock:
        if mode not in _pools:
            _pools[mode] = PooledDB(
                creator=pymysql,
                mincached=0,
                maxcached=4,
                maxconnections=4,
                maxshared=0,
                blocking=False,
                reset=True,
                ping=1,
                **connection_options(mode),
            )
        return _pools[mode]


def close_pools() -> None:
    """Close idle connections; call after changing environment configuration."""
    with _pool_lock:
        for pool in _pools.values():
            pool.close()
        _pools.clear()


@contextmanager
def connection(mode: Mode = "eval"):
    """Borrow a connection, commit on success, rollback on failure, return it."""
    conn = get_pool(mode).connection(shareable=False)
    try:
        # DBUtils must know the transaction started to avoid replaying queries
        # after a connection failure. DDL still follows MySQL implicit commits.
        conn.begin()
        yield conn
        conn.commit()
    except BaseException:
        with suppress(pymysql.Error):
            conn.rollback()
        raise
    finally:
        conn.close()


def execute_query(query: str, params=None, *, mode: Mode = "eval") -> list[dict]:
    """Run one parameterized statement, returning rows (or [] for non-SELECT).

    PyMySQL errors propagate unchanged: ``exc.args[0]`` is the MySQL error code.
    Eval permissions are enforced by MySQL, including for model-generated SQL.
    """
    with connection(mode) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            return list(cursor.fetchall()) if cursor.description else []
