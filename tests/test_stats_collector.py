import unittest
import os
import json
import time
from pathlib import Path
from raylight.utils.stats_collector import StatsCollector

class TestStatsCollector(unittest.TestCase):
    def setUp(self):
        self.collector = StatsCollector()
        self.collector.reset()
        # Use a temporary output directory for testing
        self.collector.output_dir = Path("output/test_stats")
        self.collector.output_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.collector.reset()
        # Clean up test files
        if self.collector.output_dir.exists():
            for f in self.collector.output_dir.glob("*.json"):
                f.unlink()
            self.collector.output_dir.rmdir()

    def test_singleton(self):
        s1 = StatsCollector()
        s2 = StatsCollector()
        self.assertIs(s1, s2)
        
        s1.run_id = "test_run"
        self.assertEqual(s2.run_id, "test_run")

    def test_basic_flow(self):
        config = {"model": "test_model", "steps": 10}
        self.collector.start_run(config)
        
        self.assertIsNotNone(self.collector.run_id)
        self.assertEqual(self.collector.config["model"], "test_model")
        
        self.collector.start_timer("test_timer")
        time.sleep(0.1)
        duration = self.collector.stop_timer("test_timer")
        
        self.assertGreater(duration, 0.09)
        self.assertIn("test_timer", self.collector.timings)
        
        self.collector.record_metric("memory", "peak_vram", 10.5)
        self.collector.record_metric("custom", "key", "value")
        
        self.collector.stop_run()
        
        # Verify output file
        files = list(self.collector.output_dir.glob("*.json"))
        self.assertEqual(len(files), 1)
        
        with open(files[0], 'r') as f:
            data = json.load(f)
            self.assertEqual(data["run_id"], self.collector.run_id)
            self.assertEqual(data["config"]["model"], "test_model")
            self.assertIn("test_timer", data["timings"])
            self.assertEqual(data["memory"]["peak_vram"], 10.5)
            self.assertEqual(data["config"]["custom"]["key"], "value")

    def test_update_config(self):
        self.collector.start_run({"a": 1})
        self.collector.update_config({"b": 2})
        self.assertEqual(self.collector.config["a"], 1)
        self.assertEqual(self.collector.config["b"], 2)

if __name__ == '__main__':
    unittest.main()
