import time
import unittest
from unittest.mock import patch
from app import app
from main import (
    get_steam_app_id,
    scrape_steam_reviews,
    summarize_top_reviews_gemini,
    format_bullet_points,
    get_game_reviews_and_summary,
    _analysis_cache
)


class TestSteamRecommender(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        _analysis_cache.clear()

    def test_get_steam_app_id(self):
        app_id_terraria = get_steam_app_id("Terraria")
        self.assertEqual(app_id_terraria, "105600")
        app_id_portal = get_steam_app_id("Portal 2")
        self.assertTrue(app_id_portal and app_id_portal.isdigit())

    def test_format_bullet_points(self):
        raw = "POSITIVE:\n- Great combat\n* Amazing music\n• Epic bosses"
        formatted = format_bullet_points(raw)
        self.assertIn("• Great combat", formatted)
        self.assertIn("• Amazing music", formatted)
        self.assertIn("• Epic bosses", formatted)
        self.assertNotIn("POSITIVE:", formatted)

    def test_scrape_steam_reviews(self):
        df = scrape_steam_reviews("105600", num_reviews=40)
        self.assertFalse(df.empty)
        self.assertIn("review", df.columns)
        self.assertIn("recommend", df.columns)
        self.assertIn("votes_up", df.columns)
        pos = df[df["recommend"] == True]
        neg = df[df["recommend"] == False]
        self.assertGreater(len(pos), 0)
        self.assertGreater(len(neg), 0)

    @patch("main.generate_with_gemini_fallback")
    def test_summarize_top_reviews_gemini_mocked(self, mock_gemini):
        mock_gemini.return_value = "POSITIVE:\n- Fun boss battles\n- Great progression\n\nNEGATIVE:\n- Slow early game\n- Wiki is mandatory"
        df = scrape_steam_reviews("105600", num_reviews=20)
        res = summarize_top_reviews_gemini(df)
        self.assertIn("Fun boss battles", res["positive_summary"])
        self.assertIn("Slow early game", res["negative_summary"])
        # Verify single API call was made
        self.assertEqual(mock_gemini.call_count, 1)

    @patch("main.generate_with_gemini_fallback")
    def test_caching_behavior(self, mock_gemini):
        mock_gemini.return_value = "POSITIVE:\n- Fast gameplay\n\nNEGATIVE:\n- High difficulty"
        t0 = time.time()
        res1 = get_game_reviews_and_summary("105600")
        t1 = time.time()
        self.assertEqual(mock_gemini.call_count, 1)

        # Second lookup should hit cache and NOT call Gemini again
        t2 = time.time()
        res2 = get_game_reviews_and_summary("105600")
        t3 = time.time()
        self.assertEqual(mock_gemini.call_count, 1)  # Still 1
        self.assertEqual(res1, res2)
        self.assertLess(t3 - t2, 0.05)

    @patch("app.get_game_reviews_and_summary")
    def test_api_recommendations_route(self, mock_summary):
        mock_summary.return_value = {
            "positive_summary": "• Great combat",
            "negative_summary": "• Tough puzzles"
        }
        response = self.client.get("/api/recommendations?game=Terraria")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["positive_summary"], "• Great combat")
        self.assertEqual(data["negative_summary"], "• Tough puzzles")

    def test_api_recommendations_missing_param(self):
        response = self.client.get("/api/recommendations")
        self.assertEqual(response.status_code, 400)

    def test_api_recommendations_not_found(self):
        response = self.client.get("/api/recommendations?game=xyzzy9999notarealgame")
        self.assertEqual(response.status_code, 404)

    @patch("app.get_game_reviews_and_summary")
    def test_web_index_post(self, mock_summary):
        mock_summary.return_value = {
            "positive_summary": "• Engaging mechanics",
            "negative_summary": "• Minor glitches"
        }
        response = self.client.post("/", data={"game_name": "Terraria"})
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("What players love", html)
        self.assertIn("What players criticize", html)
        self.assertIn("Engaging mechanics", html)


if __name__ == "__main__":
    unittest.main()
