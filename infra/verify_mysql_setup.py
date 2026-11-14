"""Run as python -m infra.verify_mysql_setup {init-admin,test-read,test-modal-gpu}."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pymysql

from infra.db_client import (
    ROOT, close_pools, execute_query, load_environment, required_env,
)


PROBE = "sandbox_db.__sqlgen_drop_probe"
SECRET_NAME = "mysql-eval-secret"


def check_eval_grants(rows: list[dict], *, partial_revokes: bool = False) -> None:
    """Refuse broader privileges/roles; never revoke another user's grants."""
    for row in rows:
        grant = next(iter(row.values()))
        match = re.fullmatch(r"GRANT (.+?) ON (.+?) TO .+", grant)
        if not match or "WITH GRANT OPTION" in grant:
            raise RuntimeError("Eval account has broader grants or roles; review SHOW GRANTS as admin.")
        privileges = set(match[1].split(", "))
        target = match[2]
        if privileges == {"USAGE"} and target == "*.*":
            continue
        # Unescaped '_' is a database wildcard when partial_revokes is off.
        database = "sandbox_db" if partial_revokes else r"sandbox\_db"
        # Aiven can enable ANSI_QUOTES, so SHOW GRANTS may use double quotes.
        expected = {f"`{database}`.*", f'"{database}".*', f"{database}.*"}
        if target not in expected:
            raise RuntimeError("Eval account has grants outside the exact sandbox database.")
        if not privileges <= {"SELECT", "SHOW VIEW"}:
            raise RuntimeError("Eval account has write privileges; review SHOW GRANTS as admin.")


def init_admin(*, flush_privileges: bool = False) -> dict:
    load_environment()
    user = required_env("MYSQL_EVAL_USER")
    password = required_env("MYSQL_EVAL_PASS")
    if user == required_env("MYSQL_ADMIN_USER"):
        raise ValueError("Admin and eval must be different MySQL accounts")
    # Parameters are escaped by the driver, including account names/passwords.
    execute_query(
        "CREATE USER IF NOT EXISTS %s@'%%' IDENTIFIED BY %s REQUIRE SSL",
        (user, password), mode="admin",
    )
    grants = execute_query("SHOW GRANTS FOR %s@'%%'", (user,), mode="admin")
    partial_revokes = execute_query("SELECT @@partial_revokes AS enabled", mode="admin")[0]["enabled"]
    check_eval_grants(grants, partial_revokes=bool(partial_revokes))
    execute_query("ALTER USER %s@'%%' REQUIRE SSL", (user,), mode="admin")
    execute_query("CREATE DATABASE IF NOT EXISTS sandbox_db CHARACTER SET utf8mb4", mode="admin")
    execute_query(
        "CREATE TABLE IF NOT EXISTS sandbox_db.team_check "
        "(id INT PRIMARY KEY, message VARCHAR(255) NOT NULL)", mode="admin",
    )
    execute_query(
        "INSERT INTO sandbox_db.team_check (id, message) VALUES (%s, %s) "
        "ON DUPLICATE KEY UPDATE id = id",
        (1, "sqlgen MySQL connectivity check"), mode="admin",
    )
    execute_query(f"CREATE TABLE IF NOT EXISTS {PROBE} (id INT PRIMARY KEY)", mode="admin")
    database = "sandbox_db" if partial_revokes else r"sandbox\_db"
    execute_query(
        f"GRANT SELECT, SHOW VIEW ON `{database}`.* TO %s@'%%'", (user,), mode="admin",
    )
    if flush_privileges:
        execute_query("FLUSH PRIVILEGES", mode="admin")
    return {"initialized": True, "database": "sandbox_db", "table": "team_check"}


def test_read() -> dict:
    # Both CLI paths call this same function with eval credentials only.
    cipher = execute_query("SHOW SESSION STATUS LIKE 'Ssl_cipher'")
    if not cipher or not cipher[0]["Value"]:
        raise RuntimeError("MySQL connection is not encrypted")
    partial_revokes = execute_query("SELECT @@partial_revokes AS enabled")[0]["enabled"]
    check_eval_grants(
        execute_query("SHOW GRANTS FOR CURRENT_USER()"), partial_revokes=bool(partial_revokes),
    )
    rows = execute_query("SELECT * FROM sandbox_db.team_check ORDER BY id")
    if not rows:
        raise RuntimeError("team_check is empty; run init-admin first")
    # Check it exists before DROP: a missing table is not permission evidence.
    execute_query(f"SELECT id FROM {PROBE} LIMIT 1")
    try:
        execute_query(f"DROP TABLE {PROBE}")
    except pymysql.Error as exc:
        if exc.args[0] != 1142:  # ER_TABLEACCESS_DENIED_ERROR
            raise
        return {
            "passed": True, "rows": rows, "tls_cipher": cipher[0]["Value"],
            "drop_blocked": True, "mysql_error_code": exc.args[0],
        }
    raise RuntimeError("Eval DROP unexpectedly succeeded on the disposable probe; permissions are unsafe.")


def test_modal_gpu(*, environment: str, secret_name: str) -> dict:
    import modal

    # Use a locked, small image: the connectivity check needs no CUDA Python
    # libraries. nvidia-smi verifies that a physical GPU was allocated.
    image = (
        modal.Image.debian_slim(python_version="3.13.15")
        .uv_sync(
            str(ROOT), extra_options="--locked --no-default-groups",
            frozen=False, uv_version="0.13.7",
            env={"UV_DEFAULT_INDEX": "https://pypi.org/simple", "UV_INDEX_URL": "https://pypi.org/simple"},
        )
        .add_local_python_source("infra")
    )
    app = modal.App("sqlgen-mysql-check")
    secret = modal.Secret.from_name(
        secret_name,
        required_keys=["MYSQL_HOST", "MYSQL_PORT", "MYSQL_EVAL_USER", "MYSQL_EVAL_PASS", "MYSQL_SSL_CA_PEM"],
    )

    @app.function(
        image=image, secrets=[secret], gpu=["L4", "A10"],
        cpu=1, memory=1024, timeout=120, startup_timeout=180,
        scaledown_window=2, max_containers=1, serialized=True,
    )
    def verify() -> dict:
        # No local .env or certificate file is included in the image.
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        if len(gpu) != 1 or not any(name in gpu[0] for name in ("L4", "A10")):
            raise RuntimeError("Expected exactly one L4 or A10 GPU")
        try:
            return {**test_read(), "gpu": gpu[0]}
        except pymysql.Error as exc:
            raise RuntimeError(f"Remote MySQL check failed with code {exc.args[0]}") from None
        finally:
            close_pools()

    with modal.enable_output(), app.run(environment_name=environment):
        call = verify.spawn()
        completed = False
        try:
            result = call.get(timeout=240)
            completed = True
            return result
        finally:
            if not completed:
                call.cancel(terminate_containers=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ca", type=Path, help="Aiven CA certificate (or set MYSQL_SSL_CA)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init-admin", help="Create sandbox and grant the eval account read access")
    init.add_argument("--flush-privileges", action="store_true", help="Optional; GRANT takes effect immediately")
    subparsers.add_parser("test-read", help="Verify eval reads and a permission-denied DROP")
    gpu = subparsers.add_parser("test-modal-gpu", help="Run test-read on one Modal L4/A10")
    gpu.add_argument("--profile", help="Local Modal profile for the shared workspace")
    gpu.add_argument("--env", default="main")
    gpu.add_argument("--secret-name", default=SECRET_NAME)
    args = parser.parse_args(argv)
    if args.ca:
        os.environ["MYSQL_SSL_CA"] = str(args.ca.expanduser().resolve())
    try:
        if args.command == "init-admin":
            result = init_admin(flush_privileges=args.flush_privileges)
        elif args.command == "test-read":
            result = test_read()
        else:
            if args.profile:
                # Set before importing Modal; do not rewrite the user's profile.
                os.environ["MODAL_PROFILE"] = args.profile
            result = test_modal_gpu(environment=args.env, secret_name=args.secret_name)
        print(json.dumps(result, indent=2, default=str))
        return 0
    except pymysql.Error as exc:
        print(f"MySQL error {exc.args[0]}; check TLS/CA, credentials, network access and grants.", file=sys.stderr)
        return 1
    except (ValueError, RuntimeError, OSError) as exc:
        # Config/runtime messages contain no SQL or credential values.
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        close_pools()


if __name__ == "__main__":
    raise SystemExit(main())
