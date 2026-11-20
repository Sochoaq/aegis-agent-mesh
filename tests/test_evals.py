import unittest

from aegis.evals import run_benchmark


class TestBenchmarkSuite(unittest.TestCase):
    def test_15_scenario_golden_and_red_team_eval(self):
        report = run_benchmark(output_path=None, ledger=None)
        self.assertEqual(report["total_scenarios"], 15)
        self.assertEqual(report["operational_scenarios"], 10)
        self.assertEqual(report["adversarial_scenarios"], 5)
        self.assertGreaterEqual(report["rca_runbook_top1_accuracy"], 0.90)
        self.assertGreaterEqual(report["self_reflection_triggers"], 2)
        self.assertEqual(report["grounding_citation_rate"], 1.0)
        self.assertEqual(report["adversarial_injection_block_rate"], 1.0)
        self.assertEqual(report["policy_compliance_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
