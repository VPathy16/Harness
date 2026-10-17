"""Bounded symbolic search; neural scores change order, never legal transitions."""
from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import asdict, dataclass

from .domain import Task, expand, initial, state_key, verify


@dataclass(frozen=True)
class Budget:
    max_expansions: int = 10000
    max_seconds: float = 10.0
    max_ranker_calls: int = 0

    def __post_init__(self):
        if self.max_expansions < 0 or not math.isfinite(self.max_seconds) or self.max_seconds <= 0 or self.max_ranker_calls < 0:
            raise ValueError("Invalid budget")


@dataclass
class SearchResult:
    task_id: str
    status: str
    expression: str | None
    expanded: int
    generated: int
    ranker_calls: int
    seconds: float
    trace: list

    def to_dict(self):
        return asdict(self)


def solve(task: Task, budget: Budget = Budget(), ranker=None, proposals=()):
    start = time.perf_counter()
    trace = []
    expanded = generated = calls = 0

    def finish(status, expression=None):
        return SearchResult(task.id, status, expression, expanded, generated, calls,
                            time.perf_counter() - start, trace)

    for expression in proposals:
        if time.perf_counter() - start >= budget.max_seconds:
            return finish("budget_exhausted")
        ok, reason = verify(task, expression)
        trace.append({"event": "proposal_checked", "expression": expression, "valid": ok, "reason": reason})
        if ok:
            return finish("verified", expression)

    sequence = itertools.count()
    queue = [(0, next(sequence), initial(task))]
    seen = {state_key(initial(task))}
    while queue:
        if time.perf_counter() - start >= budget.max_seconds:
            return finish("budget_exhausted")
        _, _, state = heapq.heappop(queue)
        if len(state) == 1:
            if state[0].value == task.target:
                expression = state[0].expression
                ok, reason = verify(task, expression)
                if not ok:
                    raise AssertionError(f"Search/verifier inconsistency: {reason}")
                trace.append({"event": "verified", "expression": expression})
                return finish("verified", expression)
            continue
        if expanded >= budget.max_expansions:
            return finish("budget_exhausted")
        expanded += 1
        children = []
        for child in expand(state):
            key = state_key(child)
            if key not in seen:
                seen.add(key)
                children.append(child)
        generated += len(children)
        # Score only a bounded subset. All children remain in the frontier.
        scores = {}
        if ranker is not None and calls < budget.max_ranker_calls and children:
            candidates = children[:8]
            calls += 1
            try:
                ranked = ranker.rank(task.public(), [[t.expression for t in c] for c in candidates])
                if sorted(ranked) != list(range(len(candidates))):
                    raise ValueError("Ranker returned an invalid permutation")
                scores = {index: position / max(1, len(candidates)) for position, index in enumerate(ranked)}
                trace.append({"event": "ranked", "count": len(candidates), "order": ranked})
            except Exception as exc:
                trace.append({"event": "ranker_error", "error": type(exc).__name__})
        if time.perf_counter() - start >= budget.max_seconds:
            return finish("budget_exhausted")
        for index, child in enumerate(children):
            # Depth first with deterministic tie ordering; optional neural priority.
            priority = len(child) + scores.get(index, 1.0)
            heapq.heappush(queue, (priority, next(sequence), child))
    return finish("exhausted")
