"""Exact arithmetic domain: a small executable IR and independent verifier.

This domain is deliberately narrow so success is checkable. No eval, shell,
arbitrary imports or generated Python execution is permitted.
"""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations


@dataclass(frozen=True)
class Task:
    id: str
    numbers: tuple[int, ...]
    target: int = 24

    def __post_init__(self):
        if not 2 <= len(self.numbers) <= 6:
            raise ValueError("Supply 2 to 6 integers")
        if any(type(n) is not int or not 1 <= n <= 100 for n in self.numbers):
            raise ValueError("Numbers must be positive integers <= 100")
        if type(self.target) is not int or abs(self.target) > 10000:
            raise ValueError("Target must be an integer with absolute value <= 10000")

    def public(self):
        return {"id": self.id, "numbers": list(self.numbers), "target": self.target,
                "rules": "Use every given number exactly once, only binary + - * / and parentheses."}


@dataclass(frozen=True)
class Term:
    value: Fraction
    expression: str


def initial(task: Task):
    return tuple(Term(Fraction(n), str(n)) for n in task.numbers)


def state_key(state):
    return tuple(sorted(t.value for t in state))


def expand(state):
    """Complete pairwise arithmetic expansion, modulo value-equivalent states.

    Value equivalence is valid ONLY for this domain: all initial numbers are
    always consumed exactly once, and there are no operator/shape restrictions.
    """
    seen = set()
    for i, j in combinations(range(len(state)), 2):
        a, b = state[i], state[j]
        rest = tuple(t for k, t in enumerate(state) if k not in (i, j))
        operations = [(a, b, "+", a.value + b.value), (a, b, "*", a.value * b.value),
                      (a, b, "-", a.value - b.value), (b, a, "-", b.value - a.value)]
        if b.value:
            operations.append((a, b, "/", a.value / b.value))
        if a.value:
            operations.append((b, a, "/", b.value / a.value))
        for left, right, op, value in operations:
            child = rest + (Term(value, f"({left.expression}{op}{right.expression})"),)
            key = state_key(child)
            if key not in seen:
                seen.add(key)
                yield child


def verify(task: Task, expression: str):
    """Validate the complete arithmetic specification, using exact fractions."""
    if not isinstance(expression, str) or len(expression) > 1024:
        return False, "Expression missing or too long"
    leaves = []
    try:
        root = ast.parse(expression.strip(), mode="eval")
        if sum(1 for _ in ast.walk(root)) > 64:
            return False, "Expression too complex"

        def visit(node):
            if isinstance(node, ast.Constant) and type(node.value) is int and 1 <= node.value <= 100:
                leaves.append(node.value)
                return Fraction(node.value)
            if not isinstance(node, ast.BinOp) or not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                raise ValueError("Only integer leaves and binary + - * / are allowed")
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            return a / b

        value = visit(root.body)
        if Counter(leaves) != Counter(task.numbers):
            return False, "Must use each supplied number exactly once"
        return (True, "Verified with exact arithmetic") if value == task.target else (False, f"Result is {value}, target is {task.target}")
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, RecursionError) as exc:
        return False, str(exc)
