import os
import re
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

import ai_classifier
import config
import database
import app as app_module
from scripts.prepare_data import map_product, prepare
from textutils import clean_text


class TempDBCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        database.DB_PATH = Path(self.tmp.name) / "test.db"
        database.init_db(reset=True)

    def tearDown(self):
        self.tmp.cleanup()


class PriorityTests(unittest.TestCase):
    def test_urgent_language_is_urgent(self):
        p, s = ai_classifier.analyze_priority(
            "This is urgent, I will contact a lawyer immediately about this fraud")
        self.assertEqual(p, "Urgent")
        self.assertEqual(s, "Negative")

    def test_plain_message_is_low(self):
        p, _ = ai_classifier.analyze_priority("Please update my mailing address")
        self.assertEqual(p, "Low")

    def test_now_inside_other_word_is_not_urgent(self):
        p, _ = ai_classifier.analyze_priority("I do not know my account details")
        self.assertEqual(p, "Low")

    def test_long_negative_text_is_capped(self):
        text = " ".join(["terrible awful horrible worst angry furious"] * 10)
        p, _ = ai_classifier.analyze_priority(text)
        self.assertNotEqual(p, "Urgent")


class ClassifierTests(unittest.TestCase):
    def test_result_shape_and_known_category(self):
        r = ai_classifier.classify_complaint(
            "The credit bureau still shows an account on my credit report that is not mine")
        self.assertIn(r["category"], config.CATEGORIES)
        for key in ("category_confidence", "needs_review", "priority", "sentiment"):
            self.assertIn(key, r)
        self.assertEqual(r["category"], "Credit Reporting")

    def test_model_info_reports_source(self):
        self.assertIn(ai_classifier.model_info()["source"],
                      ("trained model", "sample-data fallback"))


class DatabaseTests(TempDBCase):
    def _make(self, priority="Low"):
        return database.create_complaint(
            "Asha", "a@example.com", "Some complaint text here", "Loans", 0.9,
            priority, "Neutral")

    def test_sla_and_routing_on_create(self):
        cid = self._make("Urgent")
        c = database.get_complaint(cid)
        hours = (datetime.fromisoformat(c["due_at"])
                 - datetime.fromisoformat(c["created_at"])).total_seconds() / 3600
        self.assertAlmostEqual(hours, 4, delta=0.01)
        self.assertEqual(c["assigned_to"], config.AGENT_ROUTING["Loans"])

    def test_priority_change_recalculates_due_date(self):
        cid = self._make("Low")
        before = database.get_complaint(cid)["due_at"]
        database.update_complaint(cid, priority="Urgent")
        c = database.get_complaint(cid)
        self.assertLess(c["due_at"], before)
        hours = (datetime.fromisoformat(c["due_at"])
                 - datetime.fromisoformat(c["created_at"])).total_seconds() / 3600
        self.assertAlmostEqual(hours, 4, delta=0.01)

    def test_resolve_sets_resolved_at(self):
        cid = self._make()
        database.update_complaint(cid, status="Resolved")
        self.assertIsNotNone(database.get_complaint(cid)["resolved_at"])

    def test_update_missing_complaint_returns_false(self):
        self.assertFalse(database.update_complaint(999, status="Closed"))

    def test_sql_injection_text_is_stored_as_plain_text(self):
        cid = database.create_complaint(
            "x'); DROP TABLE complaints;--", "a@example.com", "text", "Loans",
            0.9, "Low", "Neutral")
        self.assertEqual(database.get_stats()["total"], 1)
        self.assertIn("DROP TABLE", database.get_complaint(cid)["customer_name"])

    def test_stats_count_overdue(self):
        cid = self._make("Urgent")
        conn = database.get_db()
        conn.execute("UPDATE complaints SET due_at='2000-01-01T00:00:00' WHERE id=?", (cid,))
        conn.commit()
        conn.close()
        self.assertEqual(database.get_stats()["overdue"], 1)


class RouteTests(TempDBCase):
    def setUp(self):
        super().setUp()
        app_module.app.config.update(TESTING=True, CSRF_ENABLED=False)
        self.client = app_module.app.test_client()

    GOOD = {"name": "Aysha", "email": "aysha@example.com",
            "description": "A debt collector keeps calling me about a debt I already paid"}

    def login(self):
        return self.client.post("/login", data={
            "username": config.STAFF_USERNAME, "password": config.STAFF_PASSWORD})

    def test_submit_creates_complaint_and_redirects_to_tracking(self):
        r = self.client.post("/submit", data=self.GOOD)
        self.assertEqual(r.status_code, 302)
        self.assertIn("/track/1", r.headers["Location"])
        self.assertEqual(self.client.get("/track/1").status_code, 200)

    def test_submit_rejects_bad_email(self):
        r = self.client.post("/submit", data={**self.GOOD, "email": "nope"})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(database.get_stats()["total"], 0)

    def test_track_unknown_id_is_404(self):
        self.assertEqual(self.client.get("/track/12345").status_code, 404)

    def test_console_requires_login(self):
        r = self.client.get("/console")
        self.assertEqual(r.status_code, 302)
        self.assertIn("/login", r.headers["Location"])

    def test_api_requires_login(self):
        self.assertEqual(self.client.get("/api/complaints").status_code, 401)
        self.assertEqual(self.client.get("/api/stats").status_code, 401)

    def test_wrong_password_rejected(self):
        r = self.client.post("/login", data={"username": "admin", "password": "wrong"})
        self.assertEqual(r.status_code, 401)

    def test_login_then_console_and_update(self):
        self.client.post("/submit", data=self.GOOD)
        self.assertEqual(self.login().status_code, 302)
        self.assertEqual(self.client.get("/console").status_code, 200)
        self.assertEqual(self.client.get("/api/stats").json["total"], 1)
        r = self.client.post("/complaint/1/update", data={
            "status": "Resolved", "priority": "Low", "resolution_notes": "done"})
        self.assertEqual(r.status_code, 302)
        self.assertEqual(database.get_complaint(1)["status"], "Resolved")

    def test_update_rejects_invalid_values(self):
        self.client.post("/submit", data=self.GOOD)
        self.login()
        r = self.client.post("/complaint/1/update",
                             data={"status": "Hacked", "priority": "Low"})
        self.assertEqual(r.status_code, 400)

    def test_login_next_redirect_cannot_leave_site(self):
        r = self.client.post("/login?next=//evil.com", data={
            "username": config.STAFF_USERNAME, "password": config.STAFF_PASSWORD})
        self.assertNotIn("evil.com", r.headers["Location"])

    def test_classify_preview_api(self):
        r = self.client.post("/api/classify-preview",
                             json={"description": self.GOOD["description"]})
        self.assertTrue(r.json["ready"])
        self.assertIn(r.json["category"], config.CATEGORIES)

    def test_csrf_blocks_post_without_token(self):
        app_module.app.config["CSRF_ENABLED"] = True
        r = self.client.post("/submit", data=self.GOOD)
        self.assertEqual(r.status_code, 400)

    def test_csrf_allows_post_with_token(self):
        app_module.app.config["CSRF_ENABLED"] = True
        html = self.client.get("/submit").get_data(as_text=True)
        token = re.search(r'name="_csrf" value="([^"]+)"', html).group(1)
        r = self.client.post("/submit", data={**self.GOOD, "_csrf": token})
        self.assertEqual(r.status_code, 302)


class DataPrepTests(unittest.TestCase):
    def test_product_mapping(self):
        self.assertEqual(map_product("Mortgage"), "Mortgage")
        self.assertEqual(map_product("Credit card or prepaid card"), "Credit & Prepaid Cards")
        self.assertEqual(map_product(
            "Credit reporting, credit repair services, or other personal consumer reports"),
            "Credit Reporting")
        self.assertEqual(map_product("Checking or savings account"), "Bank Accounts & Services")
        self.assertEqual(map_product(
            "Payday loan, title loan, personal loan, or advance loan"), "Loans")
        self.assertIsNone(map_product("Something else entirely"))

    def test_clean_text_removes_redactions(self):
        self.assertEqual(clean_text("I paid XXXX {$250.00} on XX/XX/2020 to Bank"),
                         "I paid on to Bank")

    def test_prepare_reads_cfpb_style_csv(self):
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / "raw.csv"
            pd.DataFrame({
                "Date received": ["2024-01-01"] * 4,
                "Product": ["Mortgage", "Debt collection", "Mortgage", "Unknown"],
                "Consumer complaint narrative": [
                    "my mortgage servicer lost my payment and charged late fees every month",
                    "collector keeps calling my phone about a debt I do not owe at all",
                    None,
                    "this row has an unmapped product and should be dropped entirely ok",
                ],
            }).to_csv(raw, index=False)
            out = prepare(raw, per_class=10)
        self.assertEqual(sorted(out["category"]), ["Debt Collection", "Mortgage"])


if __name__ == "__main__":
    unittest.main()
