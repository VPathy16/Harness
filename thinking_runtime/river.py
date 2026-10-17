"""River-crossing planning with bitset BFS and a separate set-based verifier.

No solution moves are embedded. BFS enumerates legal passenger combinations.
The verifier deliberately does not call the planner's safety or transition code.
"""
from collections import deque
from dataclasses import dataclass
from itertools import combinations
import math
import time


@dataclass(frozen=True)
class RiverTask:
    pairs: int = 5
    capacity: int = 3

    def __post_init__(self):
        if type(self.pairs) is not int or not 1 <= self.pairs <= 10:
            raise ValueError("pairs must be an integer in 1..10")
        if type(self.capacity) is not int or not 1 <= self.capacity <= 2 * self.pairs:
            raise ValueError("capacity must be an integer in 1..2*pairs")

    @property
    def people(self):
        return tuple([f"a{i}" for i in range(1, self.pairs + 1)] +
                     [f"A{i}" for i in range(1, self.pairs + 1)])

    def public(self):
        return {"pairs": self.pairs, "capacity": self.capacity,
                "people": list(self.people), "start": "Everyone and boat on left bank",
                "goal": "Everyone on right bank",
                "rules": ["a1 is paired with A1, a2 with A2, and so on.",
                          "Each crossing transports 1 to capacity people from the boat's bank.",
                          "Boat alternates banks, starting left to right; no empty crossing.",
                          "On either bank and inside the boat, an actor ai may be with another agent Aj only when Ai is also present."]}


def verify_river(task, moves):
    """Replay every move independently, checking boat, departures and arrivals."""
    left, right = set(task.people), set()
    boat_left = True
    trace = []

    def fail(reason):
        return {"valid": False, "reason": reason, "trace": trace}

    def safe(group):
        agents = {p for p in group if p.startswith("A")}
        return all("A" + p[1:] in agents or not agents
                   for p in group if p.startswith("a"))

    if not isinstance(moves, list) or len(moves) > 10000:
        return fail("moves must be a list with at most 10000 crossings")
    for index, passengers in enumerate(moves, 1):
        if not isinstance(passengers, list) or not 1 <= len(passengers) <= task.capacity:
            return fail(f"Move {index}: invalid boat occupancy")
        if any(not isinstance(p, str) or p not in task.people for p in passengers):
            return fail(f"Move {index}: unknown passenger")
        group = set(passengers)
        if len(group) != len(passengers):
            return fail(f"Move {index}: duplicate passenger")
        origin, destination = (left, right) if boat_left else (right, left)
        if not group <= origin:
            return fail(f"Move {index}: passenger absent from departure bank")
        if not safe(group):
            return fail(f"Move {index}: unsafe boat")
        origin.difference_update(group)
        if not safe(origin) or not safe(destination):
            return fail(f"Move {index}: unsafe bank during crossing")
        destination.update(group)
        if not safe(left) or not safe(right):
            return fail(f"Move {index}: unsafe bank after arrival")
        trace.append({"move": index, "direction": "L->R" if boat_left else "R->L",
                      "passengers": passengers, "left": sorted(left), "right": sorted(right)})
        boat_left = not boat_left
    if left or right != set(task.people):
        return fail("Incomplete: not everyone reached the right bank")
    return {"valid": True, "reason": "Every crossing and final state verified", "trace": trace}


def solve_river(task, max_seconds=10.0, max_expansions=1000000):
    if not math.isfinite(max_seconds) or max_seconds <= 0 or max_expansions < 0:
        raise ValueError("Invalid search budget")
    started = time.perf_counter()
    n = task.pairs
    full = (1 << (2 * n)) - 1
    low = (1 << n) - 1

    def safe(mask):
        actors, agents = mask & low, mask >> n
        return agents == 0 or actors & ~agents == 0

    start = (full, 0)
    parents = {start: None}
    queue = deque([start])
    expanded = candidates = 0

    def result(status, moves=None):
        return {"status": status, "moves": moves, "expanded": expanded,
                "candidate_crossings": candidates, "visited_states": len(parents),
                "seconds": time.perf_counter() - started,
                "shortest_path": status == "verified"}

    while queue:
        if time.perf_counter() - started >= max_seconds:
            return result("budget_exhausted")
        state = queue.popleft()
        left, side = state
        if left == 0:
            moves = []
            current = state
            while parents[current] is not None:
                previous, mask = parents[current]
                moves.append([p for i, p in enumerate(task.people) if mask & (1 << i)])
                current = previous
            moves.reverse()
            verified = verify_river(task, moves)
            if not verified["valid"]:
                raise AssertionError(verified["reason"])
            if time.perf_counter() - started >= max_seconds:
                return result("budget_exhausted")
            return result("verified", moves)
        if expanded >= max_expansions:
            return result("budget_exhausted")
        expanded += 1
        available = left if side == 0 else full ^ left
        indices = [i for i in range(2 * n) if available & (1 << i)]
        for size in range(1, min(task.capacity, len(indices)) + 1):
            for people in combinations(indices, size):
                candidates += 1
                if time.perf_counter() - started >= max_seconds:
                    return result("budget_exhausted")
                mask = sum(1 << i for i in people)
                new_left = left ^ mask
                if not safe(mask) or not safe(new_left) or not safe(full ^ new_left):
                    continue
                child = (new_left, 1 - side)
                if child not in parents:
                    parents[child] = (state, mask)
                    queue.append(child)
    return result("exhausted")
