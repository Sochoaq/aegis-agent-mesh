import unittest

from aegis.rag import DEFAULT_RUNBOOKS, HybridRAGEngine


class TestHybridRAGEngine(unittest.TestCase):
    def setUp(self):
        self.engine = HybridRAGEngine()

    def test_all_six_runbooks_indexed_and_rank_accurately(self):
        self.assertEqual(len(self.engine.runbooks), 6)
        queries = {
            "RB-101": "PgBouncer idle in transaction connection pool lock timeout",
            "RB-102": "Kubernetes pod OOMKilled cgroup exit code 137 memory leak heap",
            "RB-103": "ESP32 edge gateway BLE MQTT CRC packet loss telemetry broker",
            "RB-104": "API gateway 504 upstream timeout P99 latency worker thread saturation",
            "RB-105": "IAM JWT 401 unauthorized credential stuffing API key anomaly",
            "RB-106": "PyTorch model serving PSI concept drift feature skew calibration",
        }
        for expected_id, q in queries.items():
            with self.subTest(expected_id=expected_id):
                hits = self.engine.search(q, top_k=3)
                self.assertEqual(len(hits), 3)
                self.assertEqual(hits[0].runbook_id, expected_id)
                self.assertGreater(hits[0].rrf_score, 0.0)
                self.assertGreater(hits[0].bm25_score, 0.0)
                self.assertGreater(hits[0].cosine_score, 0.0)

    def test_rrf_score_monotonic_order(self):
        hits = self.engine.search("postgres connection pool lock", top_k=len(DEFAULT_RUNBOOKS))
        scores = [h.rrf_score for h in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main()
