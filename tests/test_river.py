import unittest
from thinking_runtime.river import RiverTask, solve_river, verify_river


class RiverTests(unittest.TestCase):
    def test_known_shortest_lengths(self):
        for pairs, length in [(2, 5), (3, 11)]:
            task = RiverTask(pairs, 2)
            result = solve_river(task)
            self.assertEqual(result["status"], "verified")
            self.assertEqual(len(result["moves"]), length)
            self.assertTrue(verify_river(task, result["moves"])["valid"])

    def test_impossibility_distinguished_from_budget(self):
        self.assertEqual(solve_river(RiverTask(4, 2))["status"], "exhausted")
        self.assertEqual(solve_river(RiverTask(), max_expansions=0)["status"], "budget_exhausted")

    def test_unsafe_boat_even_when_destination_would_be_safe(self):
        result = verify_river(RiverTask(2, 2), [["a1", "A2"]])
        self.assertIn("unsafe boat", result["reason"])

    def test_unsafe_departure_bank(self):
        self.assertIn("unsafe bank", verify_river(RiverTask(2, 2), [["A1"]])["reason"])

    def test_invalid_and_incomplete_moves(self):
        task = RiverTask(2, 2)
        for moves in [[], [[]], [["a1", "a1"]], [["x"]], [["a1", "a2", "A1"]],
                      [["a1", "a2"], ["A1"]], "not a list"]:
            self.assertFalse(verify_river(task, moves)["valid"], moves)

    def test_five_pair_solution_and_corruption(self):
        task = RiverTask()
        moves = solve_river(task)["moves"]
        self.assertTrue(verify_river(task, moves)["valid"])
        for i in range(len(moves)):
            corrupted = [list(m) for m in moves]
            corrupted[i] = []
            self.assertFalse(verify_river(task, corrupted)["valid"])


if __name__ == "__main__":
    unittest.main()
