"""Local authenticated schema-search API; run behind a proxy for deployment.

One server credential represents ONE trusted application, not individual users.
This intentionally serial HTTP service fits a learning demo, not public traffic.
Use TLS, identity-aware authorization and a managed server before exposing it.
"""
import argparse
import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import sys
import time
import uuid

from rag.bm25 import BM25Index


def make_handler(index: BM25Index, token: str, allowed: set[str]):
    """Bind trusted server configuration; clients cannot supply their allowlist.

    index: Verified snapshot loaded once, not rebuilt for every question.
    token: Secret obtained from the environment; never passed in URLs/logs.
    allowed: Databases accessible to this service's single application identity.
    """
    if len(token) < 32 or not allowed or not allowed <= index.by_db.keys():
        raise ValueError("Require a >=32-character token and known permitted databases")

    class Handler(BaseHTTPRequestHandler):
        server_version = "SchemaSearch/1"

        def setup(self):
            super().setup()
            self.connection.settimeout(5)  # Bound a slow local client's socket reads.

        def log_message(self, *args):
            # Default request logging can expose URLs. We emit our own compact
            # POST event without the question, credentials, SQL, or schema text.
            pass

        def respond(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.respond(200, {"status": "ready"})
            else:
                self.respond(404, {"error": "not_found"})

        def do_POST(self):
            request_id = uuid.uuid4().hex
            started = time.perf_counter()
            status = 500
            try:
                if self.path != "/search":
                    status = 404
                    self.respond(status, {"error": "not_found", "request_id": request_id})
                    return
                if not hmac.compare_digest(self.headers.get("Authorization", "").encode(),
                                           ("Bearer " + token).encode()):
                    status = 401
                    self.respond(status, {"error": "unauthorized", "request_id": request_id})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if self.headers.get("Transfer-Encoding") or not 0 < length <= 16384:
                    status = 413
                    self.respond(status, {"error": "invalid_body_size", "request_id": request_id})
                    return
                if self.headers.get_content_type() != "application/json":
                    status = 415
                    self.respond(status, {"error": "json_required", "request_id": request_id})
                    return
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict) or set(body) - {"query", "k", "db_id", "explain"}:
                    raise ValueError("Unexpected request fields")
                if not isinstance(body.get("explain", False), bool):
                    raise ValueError("explain must be boolean")
                if body.get("db_id") is not None and not isinstance(body["db_id"], str):
                    raise ValueError("db_id must be a string")
                result = index.search(body.get("query"), k=body.get("k", 5), db_id=body.get("db_id"),
                                      allowed_db_ids=allowed, explain=body.get("explain", False))
                status = 200
                self.respond(status, {"request_id": request_id, **result})
            except (ValueError, UnicodeError, RecursionError):
                status = 400
                self.respond(status, {"error": "invalid_request_or_scope", "request_id": request_id})
            except TimeoutError:
                status = 408
                self.respond(status, {"error": "request_timeout", "request_id": request_id})
            finally:
                print(json.dumps({"request_id": request_id, "status": status,
                                  "elapsed_ms": (time.perf_counter() - started) * 1000}), file=sys.stderr)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--allow-db", action="append", required=True,
                        help="Server-controlled database; repeat for multiple databases")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    index = BM25Index.load(args.index, args.bundle)
    handler = make_handler(index, os.environ.get("RAG_API_TOKEN", ""), set(args.allow_db))
    # Loopback only: this service intentionally cannot bind a public interface.
    with HTTPServer(("127.0.0.1", args.port), handler) as server:
        print(json.dumps({"listening": f"http://127.0.0.1:{args.port}", "tables": len(index.documents)}), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
