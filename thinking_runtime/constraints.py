"""Small domain-independent integer constraint IR and bounded backtracking.

No scheduling/job/warehouse names or task-specific answers appear here.
Generated text is data; it is never executed as Python.
"""
from itertools import product
import math
import time

OPS = ("all_different", "eq", "neq", "lt", "offset", "sum_eq")
IR_SCHEMA = {
    "type": "object", "properties": {
        "variables": {"type": "array", "minItems": 1, "maxItems": 10, "items": {
            "type": "object", "properties": {
                "name": {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_]{0,15}$"},
                "domain": {"type": "array", "minItems": 1, "maxItems": 12,
                           "items": {"type": "integer", "minimum": -100, "maximum": 100}}},
            "required": ["name", "domain"], "additionalProperties": False}},
        "constraints": {"type": "array", "maxItems": 50, "items": {
            "type": "object", "properties": {
                "op": {"type": "string", "enum": list(OPS)},
                "vars": {"type": "array", "minItems": 1, "maxItems": 10,
                         "items": {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_]{0,15}$"}},
                "value": {"type": "integer", "minimum": -1000, "maximum": 1000}},
            "required": ["op", "vars", "value"], "additionalProperties": False}}},
    "required": ["variables", "constraints"], "additionalProperties": False}

LANGUAGE = """Translate the user's problem into this integer constraint language. Do not solve it.
Return JSON with variables and constraints. Preserve every rule, even contradictions.
variables: list of {name: string, domain: list of allowed integers}.
constraints: list of {op: string, vars: list of variable names, value: integer}.
Operators:
all_different: all listed variables have different values; value must be 0.
eq: exactly one variable equals value.
neq: exactly one variable differs from value.
lt: exactly two variables; the first is strictly less than the second; value is 0.
offset: exactly two variables; second = first + value.
sum_eq: sum of listed variables equals value.
Use existing variable names exactly. Include complete allowed domains. Use only these operators.
Example: 'X and Y take distinct values in 0,1,2; X is smaller than Y.' becomes
{"variables":[{"name":"X","domain":[0,1,2]},{"name":"Y","domain":[0,1,2]}],"constraints":[{"op":"all_different","vars":["X","Y"],"value":0},{"op":"lt","vars":["X","Y"],"value":0}]}.
Output only the JSON object. No explanation or code.
"""


def validate(ir):
    if not isinstance(ir, dict) or set(ir) != {"variables", "constraints"}:
        raise ValueError("IR needs exactly variables and constraints")
    variables, constraints = ir["variables"], ir["constraints"]
    if not isinstance(variables, list) or not 1 <= len(variables) <= 10:
        raise ValueError("Need 1..10 variables")
    if not isinstance(constraints, list) or len(constraints) > 50:
        raise ValueError("Need at most 50 constraints")
    names = set()
    for v in variables:
        if not isinstance(v, dict) or set(v) != {"name", "domain"}:
            raise ValueError("Bad variable")
        name, domain = v["name"], v["domain"]
        if not isinstance(name, str) or not name.isascii() or not name or not name[0].isalpha() or len(name) > 16 or not name.replace('_', '').isalnum() or name in names:
            raise ValueError("Invalid or duplicate variable name")
        names.add(name)
        if not isinstance(domain, list) or not 1 <= len(domain) <= 12 or any(type(x) is not int or abs(x) > 100 for x in domain) or len(set(domain)) != len(domain):
            raise ValueError("Invalid domain")
    for c in constraints:
        if not isinstance(c, dict) or set(c) != {"op", "vars", "value"} or c["op"] not in OPS:
            raise ValueError("Invalid constraint")
        scope = c["vars"]
        if not isinstance(scope, list) or not 1 <= len(scope) <= 10 or any(not isinstance(v,str) or v not in names for v in scope) or len(set(scope)) != len(scope):
            raise ValueError("Invalid constraint references")
        if type(c['value']) is not int or abs(c['value']) > 1000:
            raise ValueError("Invalid constraint value")
        if c['op'] in ('eq', 'neq') and len(scope) != 1:
            raise ValueError("eq/neq require one variable")
        if c['op'] in ('lt', 'offset') and len(scope) != 2:
            raise ValueError("lt/offset require two variables")
        if c['op'] in ('lt', 'all_different') and c['value'] != 0:
            raise ValueError("Unused value must be zero")
    return ir


def partial_ok(constraints, assignment):
    for c in constraints:
        vals = [assignment[v] for v in c['vars'] if v in assignment]
        op, value = c['op'], c['value']
        if op == 'all_different':
            if len(vals) != len(set(vals)): return False
            continue
        if len(vals) != len(c['vars']): continue
        if op == 'eq' and vals[0] != value: return False
        if op == 'neq' and vals[0] == value: return False
        if op == 'lt' and not vals[0] < vals[1]: return False
        if op == 'offset' and vals[1] != vals[0] + value: return False
        if op == 'sum_eq' and sum(vals) != value: return False
    return True


def verify_assignment(ir, assignment):
    """Separate complete-assignment evaluator; no planner pruning is reused."""
    validate(ir)
    names = {v['name'] for v in ir['variables']}
    if not isinstance(assignment, dict) or set(assignment) != names:
        return False
    if any(type(assignment[v['name']]) is not int or assignment[v['name']] not in v['domain'] for v in ir['variables']):
        return False
    for c in ir['constraints']:
        scope, op, target = c['vars'], c['op'], c['value']
        values = tuple(assignment[name] for name in scope)
        if op == 'all_different' and any(values[i] == values[j] for i in range(len(values)) for j in range(i)): return False
        if op == 'eq' and values[0] != target: return False
        if op == 'neq' and values[0] == target: return False
        if op == 'lt' and values[0] >= values[1]: return False
        if op == 'offset' and values[1] - values[0] != target: return False
        if op == 'sum_eq' and sum(values) != target: return False
    return True


def solve_constraints(ir, seconds=10, max_nodes=1000000):
    validate(ir)
    if not math.isfinite(seconds) or seconds <= 0 or max_nodes < 0:
        raise ValueError("Invalid budget")
    start = time.perf_counter()
    # Domain size and constraint degree are task-independent ordering heuristics.
    variables = sorted(ir['variables'], key=lambda v:(len(v['domain']), -sum(v['name'] in c['vars'] for c in ir['constraints']), v['name']))
    nodes = 0
    assignment = {}

    def visit(index):
        nonlocal nodes
        if time.perf_counter() - start >= seconds or nodes >= max_nodes:
            raise TimeoutError
        nodes += 1
        if index == len(variables):
            if not verify_assignment(ir, assignment): raise AssertionError('Planner/verifier disagreement')
            return dict(assignment)
        var = variables[index]
        for value in var['domain']:
            assignment[var['name']] = value
            if partial_ok(ir['constraints'], assignment):
                result = visit(index+1)
                if result is not None: return result
        assignment.pop(var['name'], None)
        return None

    try:
        answer = visit(0)
        status = 'verified' if answer is not None else 'unsatisfiable'
    except TimeoutError:
        answer, status = None, 'budget_exhausted'
    return {'status':status,'assignment':answer,'nodes':nodes,'seconds':time.perf_counter()-start}


def semantic_audit(reference, candidate, max_assignments=200000):
    """Evaluation ONLY: exhaustive equivalence to a withheld human-authored spec.

This is not an automatic oracle for arbitrary natural-language correctness.
The gold spec is never available to the model or to its production search.
"""
    validate(reference)
    validate(candidate)
    def all_solutions(ir):
        variables = sorted(ir['variables'], key=lambda v:v['name'])
        if math.prod(len(v['domain']) for v in variables) > max_assignments:
            raise ValueError('Audit enumeration budget exceeded')
        names = tuple(v['name'] for v in variables)
        solutions = set()
        for values in product(*(v['domain'] for v in variables)):
            if verify_assignment(ir, dict(zip(names,values))): solutions.add(values)
        return names, solutions
    rn, rs = all_solutions(reference)
    cn, cs = all_solutions(candidate)
    if rn != cn:
        return {'equivalent':False,'reason':'Variable names differ'}
    return {'equivalent':rs==cs,'reference_solution_count':len(rs),'candidate_solution_count':len(cs),
            'missing_solutions':len(rs-cs),'extra_solutions':len(cs-rs)}
