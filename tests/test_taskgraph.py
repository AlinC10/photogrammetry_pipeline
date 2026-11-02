"""pipeline/taskgraph.py: dependencies, partial failure, no deadlock, critical-path order, limits, records."""

import threading
import time
import unittest

from pipeline.taskgraph import DONE, FAILED, SKIPPED, TaskError, TaskGraph


class TestTaskGraph(unittest.TestCase):
    def test_dependencies_and_results(self):
        order = []
        lock = threading.Lock()

        def step(name, value):
            with lock:
                order.append(name)
            return value

        with TaskGraph(max_workers=4) as g:
            g.add("c", step, "c", 3, deps=["a", "b"])
            g.add("a", step, "a", 1)
            g.add("b", step, "b", 2, deps=["a"])
            self.assertTrue(g.wait(timeout=5))
            self.assertEqual([g.result(n) for n in "abc"], [1, 2, 3])
        self.assertEqual(order, ["a", "b", "c"])

    def test_failure_skips_only_dependents(self):
        def boom():
            raise ValueError("bad photo")

        with TaskGraph(max_workers=2) as g:
            g.add("bad", boom)
            g.add("after_bad", lambda: 1, deps=["bad"])
            g.add("after_after", lambda: 2, deps=["after_bad"])
            g.add("independent", lambda: 3)
            self.assertTrue(g.wait(timeout=5))
            self.assertEqual(g.state("bad"), FAILED)
            self.assertEqual(g.state("after_bad"), SKIPPED)
            self.assertEqual(g.state("after_after"), SKIPPED)
            self.assertEqual(g.state("independent"), DONE)
            self.assertEqual(g.result("independent"), 3)
            with self.assertRaises(TaskError):
                g.result("bad")
            with self.assertRaises(TaskError):
                g.result("after_after")

    def test_missing_dependency_does_not_hang(self):
        with TaskGraph(max_workers=2) as g:
            g.add("x", lambda: 1, deps=["never_added"])
            g.add("y", lambda: 2)
            t0 = time.time()
            self.assertFalse(g.wait(timeout=10))
            self.assertLess(time.time() - t0, 5)
            self.assertEqual(g.result("y"), 2)
            self.assertFalse(g.wait(["x"], timeout=0.2))

    def test_tasks_added_while_running(self):
        with TaskGraph(max_workers=2) as g:
            def spawner():
                for i in range(3):
                    g.add(f"child{i}", lambda i=i: i * 10)
                return "spawned"

            g.add("spawner", spawner)
            g.add("join", lambda: "joined", deps=["spawner", "child0", "child1", "child2"])
            self.assertTrue(g.wait(timeout=5))
            self.assertEqual(g.result("join"), "joined")
            self.assertEqual(g.result("child2"), 20)

    def test_critical_path_first(self):
        """One worker: the ready task heading the longer chain starts first, whatever the order of adding."""
        order = []
        gate = threading.Event()
        with TaskGraph(max_workers=1) as g:
            g.add("gate", gate.wait, 5)
            g.add("short", order.append, "short", deps=["gate"], cost=1)
            g.add("long_head", order.append, "long_head", deps=["gate"], cost=1)
            g.add("long_tail", order.append, "long_tail", deps=["long_head"], cost=20)
            gate.set()
            self.assertTrue(g.wait(timeout=5))
        self.assertEqual(order[0], "long_head")

    def test_resource_limit(self):
        running, peak = [0], [0]
        lock = threading.Lock()

        def heavy():
            with lock:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            time.sleep(0.05)
            with lock:
                running[0] -= 1

        with TaskGraph(max_workers=4, limits={"gpu": 1}) as g:
            for i in range(4):
                g.add(f"g{i}", heavy, resource="gpu")
            self.assertTrue(g.wait(timeout=5))
        self.assertEqual(peak[0], 1)

    def test_records_overlap(self):
        with TaskGraph(max_workers=2) as g:
            g.add("a", time.sleep, 0.15)
            g.add("b", time.sleep, 0.15)
            g.add("c", lambda: None, deps=["a", "b"])
            g.wait(timeout=5)
            recs = {r["name"]: r for r in g.records()}
        self.assertIn("b", recs["a"]["overlaps_with"])
        self.assertNotIn("a", recs["c"]["overlaps_with"])
        self.assertGreaterEqual(recs["c"]["start_s"], recs["a"]["end_s"] - 0.01)
        self.assertEqual(recs["c"]["state"], DONE)

    def test_duplicate_name_rejected(self):
        with TaskGraph() as g:
            g.add("a", lambda: 1)
            with self.assertRaises(ValueError):
                g.add("a", lambda: 2)


if __name__ == "__main__":
    unittest.main()
