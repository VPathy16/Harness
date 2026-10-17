import argparse
import itertools
import tempfile
import time
import unittest
from fractions import Fraction

from thinking_runtime.domain import Task, verify
from thinking_runtime.runtime import Budget, solve
from thinking_runtime.workspace import Workspace
from thinking_runtime.cli import run_case


def independent_oracle(values, target):
    if len(values) == 1:
        return values[0] == target
    for i in range(len(values)):
        for j in range(len(values)):
            if i == j: continue
            a, b = values[i], values[j]
            rest = [v for k, v in enumerate(values) if k not in (i, j)]
            results = [a + b, a - b, a * b]
            if b: results.append(a / b)
            for result in results:
                if independent_oracle(rest + [result], target): return True
    return False


class RuntimeTests(unittest.TestCase):
    def test_exact_fractional_solution(self):
        task = Task("fraction", (3, 3, 8, 8))
        self.assertTrue(verify(task, "8/(3-8/3)")[0])
        result = solve(task)
        self.assertEqual(result.status, "verified")
        self.assertTrue(verify(task, result.expression)[0])

    def test_verifier_rejects_injection_and_number_reuse(self):
        task = Task("input", (1, 2, 3, 4))
        for invalid in ["24", "__import__('os').system('id')", "4*4+4+4", "2**3+16", "True*24", "1/(2-2)"]:
            self.assertFalse(verify(task, invalid)[0], invalid)

    def test_bad_proposal_cannot_claim_success(self):
        result = solve(Task("bad", (3, 3, 8, 8)), proposals=["24"])
        self.assertFalse(result.trace[0]["valid"])
        self.assertEqual(result.status, "verified")

    def test_budget_is_not_unsatisfiable(self):
        self.assertEqual(solve(Task("cutoff", (3, 3, 8, 8)), Budget(0)).status, "budget_exhausted")
        self.assertEqual(solve(Task("impossible", (1, 1, 1, 1))).status, "exhausted")

    def test_ranker_cannot_remove_search_branches(self):
        class BrokenRanker:
            def rank(self, task, candidates): return [999]
        result = solve(Task("ranker", (3, 3, 8, 8)), Budget(max_ranker_calls=2), ranker=BrokenRanker())
        self.assertEqual(result.status, "verified")
        self.assertEqual(result.ranker_calls, 2)
        self.assertTrue(any(e["event"] == "ranker_error" for e in result.trace))

    def test_deduplication_preserves_solvability(self):
        # Compare against an independent ordered enumeration on 35 multisets.
        for numbers in itertools.combinations_with_replacement(range(1, 5), 4):
            task = Task(str(numbers), numbers)
            expected = independent_oracle([Fraction(n) for n in numbers], Fraction(24))
            result = solve(task)
            self.assertEqual(result.status == "verified", expected, numbers)
            self.assertNotEqual(result.status, "budget_exhausted")

    def test_memory_rechecks_claimed_success(self):
        with tempfile.TemporaryDirectory() as folder:
            w = Workspace(folder)
            task = Task("memory", (1, 2, 3, 4))
            w.record(task, {"status": "verified", "expression": "24"})
            self.assertIsNone(w.recall(task))
            w.record(task, {"status": "verified", "expression": "1*2*3*4"})
            self.assertEqual(w.recall(task), "1*2*3*4")
            w.close()

    def test_late_model_answer_does_not_pass_benchmark(self):
        class LateGenerator:
            def generate(self, *args):
                time.sleep(.01)
                return {"expression": "1*2*3*4", "seconds": .01, "input_tokens": None, "output_tokens": None}
        args = argparse.Namespace(mode="direct", seconds=.001, max_calls=1, expansions=100,
                    ranker_calls=0, usd_per_hour=None, backend="http",
                    input_usd_per_million=None, output_usd_per_million=None)
        row = run_case(Task("late", (1, 2, 3, 4)), args, LateGenerator(), None)
        self.assertEqual(row["status"], "budget_exhausted")
        self.assertIsNone(row["estimated_execution_usd"])


if __name__ == "__main__":
    unittest.main()
