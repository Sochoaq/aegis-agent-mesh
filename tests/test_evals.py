import unittest

from aegis.evals import run_benchmark


class TestBenchmarkSuite(unittest.TestCase):
    def test_12_scenario_golden_and_red_team_eval(self):
        report = run_benchmark(output_path=None, ledger=None)
        self.assertEqual(report["total_scenarios"], 12)
        self.assertEqual(report["operational_scenarios"], 8)
        self.assertEqual(report["adversarial_scenarios"], 4)
        self.assertGreaterEqual(report["rca_runbook_top1_accuracy"], 0.90)
        self.assertEqual(report["grounding_citation_rate"], 1.0)
        self.assertEqual(report["adversarial_injection_block_rate"], 1.0)
        self.assertEqual(report["policy_compliance_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
