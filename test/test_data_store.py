import os
import sys
import unittest
import threading
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'optimizer')))
from data_store import DataStore

class TestDataStore(unittest.TestCase):
    def setUp(self):
        self.log_file = "test_optimization_log.jsonl"
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        self.ds = DataStore(log_file=self.log_file)

    def tearDown(self):
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def test_concurrent_appends(self):
        def worker(tid):
            self.ds.append_result({
                "status": "success",
                "parameters": {"test_id": tid},
                "metrics": {"score": tid}
            })

        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        history = self.ds.load_history()
        self.assertEqual(len(history), 10)

        # Verify all thread IDs made it in
        tids = sorted([r["parameters"]["test_id"] for r in history])
        self.assertEqual(tids, list(range(10)))

    def test_missing_params(self):
        with self.assertRaisesRegex(ValueError, "Missing required field: parameters"):
            self.ds.append_result({"status": "failed"})

if __name__ == "__main__":
    unittest.main()
