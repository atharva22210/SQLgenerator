"""Permission probes must fail on unexpected errors or excessive privileges."""

import unittest
from unittest.mock import patch

import pymysql

from infra import verify_mysql_setup as setup


def grants(*statements):
    return [{"Grants": statement} for statement in statements]


class GrantTests(unittest.TestCase):
    def test_exact_read_only_database_grants_are_allowed(self):
        setup.check_eval_grants(grants(
            "GRANT USAGE ON *.* TO `reader`@`%`",
            r"GRANT SELECT, SHOW VIEW ON `sandbox\_db`.* TO `reader`@`%`",
        ))

    def test_wildcards_writes_roles_and_grant_option_are_rejected(self):
        for statement in (
            "GRANT SELECT ON *.* TO `reader`@`%`",
            "GRANT SELECT ON `sandbox_db`.* TO `reader`@`%`",
            r"GRANT SELECT, DROP ON `sandbox\_db`.* TO `reader`@`%`",
            "GRANT `admin_role`@`%` TO `reader`@`%`",
            r"GRANT SELECT ON `sandbox\_db`.* TO `reader`@`%` WITH GRANT OPTION",
        ):
            with self.subTest(statement=statement), self.assertRaises(RuntimeError):
                setup.check_eval_grants(grants(statement))

    def test_partial_revokes_treats_unescaped_database_as_literal(self):
        setup.check_eval_grants(
            grants("GRANT SELECT ON `sandbox_db`.* TO `reader`@`%`"), partial_revokes=True,
        )

    def test_ansi_quotes_in_live_aiven_show_grants(self):
        setup.check_eval_grants(
            grants('GRANT SELECT, SHOW VIEW ON "sandbox_db".* TO "reader"@"%"'),
            partial_revokes=True,
        )


class ReadProbeTests(unittest.TestCase):
    def run_probe(self, drop_error):
        def query(sql, *args, **kwargs):
            self.assertNotEqual(kwargs.get("mode"), "admin")
            if sql.startswith("SHOW SESSION"):
                return [{"Value": "TLS_AES_256_GCM_SHA384"}]
            if sql.startswith("SHOW GRANTS"):
                return grants(r"GRANT SELECT, SHOW VIEW ON `sandbox\_db`.* TO `reader`@`%`")
            if sql.startswith("SELECT @@partial"):
                return [{"enabled": 0}]
            if sql.startswith("SELECT *"):
                return [{"id": 1, "message": "test row"}]
            if sql.startswith("DROP"):
                self.assertEqual(sql, f"DROP TABLE {setup.PROBE}")
                if drop_error:
                    raise drop_error
            return []
        with patch.object(setup, "execute_query", side_effect=query):
            return setup.test_read()

    def test_only_permission_denial_counts_as_success(self):
        result = self.run_probe(pymysql.OperationalError(1142, "denied"))
        self.assertTrue(result["passed"])
        self.assertEqual(result["mysql_error_code"], 1142)

    def test_missing_table_or_lost_connection_is_not_permission_evidence(self):
        for code in (1051, 2013):
            with self.subTest(code=code), self.assertRaises(pymysql.Error):
                self.run_probe(pymysql.OperationalError(code, "test error"))

    def test_unexpected_drop_success_fails_the_check(self):
        with self.assertRaisesRegex(RuntimeError, "unexpectedly succeeded"):
            self.run_probe(None)


if __name__ == "__main__":
    unittest.main()
