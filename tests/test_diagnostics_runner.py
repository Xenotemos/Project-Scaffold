"""Diagnostics harness tests."""

from __future__ import annotations

import json
import unittest

from scripts import diagnostics


class DiagnosticsRunnerTests(unittest.TestCase):
    def test_run_diagnostics_produces_results(self) -> None:
        results = diagnostics.run_diagnostics(allow_repair=False)
        self.assertTrue(results)
        for result in results:
            self.assertIsInstance(result.label, str)
            self.assertTrue(result.label)
            self.assertIn(result.status, {diagnostics.STATUS_OK, diagnostics.STATUS_WARN, diagnostics.STATUS_ERR})
            self.assertIsInstance(result.metadata, dict)

    def test_json_format_roundtrip(self) -> None:
        results = diagnostics.run_diagnostics(allow_repair=False)
        payload = json.loads(diagnostics._format_json(results))
        self.assertIn("results", payload)
        self.assertGreater(len(payload["results"]), 0)


if __name__ == "__main__":
    unittest.main()
