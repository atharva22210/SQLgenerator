"""Offline checks for credential separation, TLS and pooled query lifecycle."""

import os
import ssl
import unittest
from unittest.mock import MagicMock, patch

import pymysql

from infra import db_client


class ConnectionOptionsTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            "MYSQL_HOST": "mysql.example.test", "MYSQL_PORT": "3306",
            "MYSQL_ADMIN_USER": "setup", "MYSQL_ADMIN_PASS": "test-admin-placeholder",
            "MYSQL_EVAL_USER": "reader", "MYSQL_EVAL_PASS": "test-eval-placeholder",
        }, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        loader = patch.object(db_client, "load_environment")
        loader.start()
        self.addCleanup(loader.stop)

    def test_eval_is_default_and_tls_verifies_identity(self):
        options = db_client.connection_options()
        self.assertEqual(options["user"], "reader")
        self.assertEqual(options["password"], "test-eval-placeholder")
        self.assertEqual(options["database"], "sandbox_db")
        self.assertTrue(options["ssl"].check_hostname)
        self.assertEqual(options["ssl"].verify_mode, ssl.CERT_REQUIRED)
        self.assertEqual(options["ssl"].minimum_version, ssl.TLSVersion.TLSv1_2)
        self.assertFalse(options["local_infile"])

    def test_admin_can_connect_before_database_exists(self):
        options = db_client.connection_options("admin")
        self.assertEqual(options["user"], "setup")
        self.assertIsNone(options["database"])

    def test_eval_does_not_need_admin_credentials(self):
        del os.environ["MYSQL_ADMIN_USER"]
        del os.environ["MYSQL_ADMIN_PASS"]
        self.assertEqual(db_client.connection_options()["user"], "reader")

    def test_invalid_mode_and_missing_password_fail_closed(self):
        with self.assertRaises(ValueError):
            db_client.connection_options("other")
        del os.environ["MYSQL_EVAL_PASS"]
        with self.assertRaisesRegex(ValueError, "MYSQL_EVAL_PASS"):
            db_client.connection_options()

    def test_invalid_ca_is_not_ignored(self):
        os.environ["MYSQL_SSL_CA_PEM"] = "not a certificate"
        with self.assertRaises(ssl.SSLError):
            db_client.connection_options()


class QueryLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.conn = MagicMock()
        self.cursor = self.conn.cursor.return_value.__enter__.return_value
        self.cursor.description = ("id",)
        self.cursor.fetchall.return_value = [{"id": 1}]
        patcher = patch.object(db_client, "get_pool")
        self.pool = patcher.start().return_value
        self.pool.connection.return_value = self.conn
        self.addCleanup(patcher.stop)

    def test_select_binds_parameters_commits_and_returns_connection(self):
        self.assertEqual(db_client.execute_query("SELECT %s AS id", (1,)), [{"id": 1}])
        self.cursor.execute.assert_called_once_with("SELECT %s AS id", (1,))
        self.conn.begin.assert_called_once()
        self.conn.commit.assert_called_once()
        self.conn.close.assert_called_once()
        self.conn.rollback.assert_not_called()

    def test_mysql_error_and_code_are_preserved_after_rollback(self):
        error = pymysql.OperationalError(1142, "test permission denied")
        self.cursor.execute.side_effect = error
        with self.assertRaises(pymysql.Error) as raised:
            db_client.execute_query("DROP TABLE test_probe")
        self.assertIs(raised.exception, error)
        self.conn.rollback.assert_called_once()
        self.conn.close.assert_called_once()
        self.conn.commit.assert_not_called()

    def test_non_select_returns_empty_rows(self):
        self.cursor.description = None
        self.assertEqual(db_client.execute_query("INSERT INTO example VALUES (%s)", (1,), mode="admin"), [])
        self.cursor.fetchall.assert_not_called()


if __name__ == "__main__":
    unittest.main()
