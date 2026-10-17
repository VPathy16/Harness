import copy
import unittest
from thinking_runtime.constraints import solve_constraints, verify_assignment, validate, semantic_audit


class ConstraintTests(unittest.TestCase):
    def setUp(self):
        self.ir = {'variables':[{'name':'X','domain':[1,2,3]},{'name':'Y','domain':[1,2,3]}],
                   'constraints':[{'op':'lt','vars':['X','Y'],'value':0},
                                  {'op':'sum_eq','vars':['X','Y'],'value':4}]}

    def test_solver_and_independent_verifier(self):
        result = solve_constraints(self.ir)
        self.assertEqual(result['assignment'],{'X':1,'Y':3})
        self.assertTrue(verify_assignment(self.ir,result['assignment']))
        for bad in [{'X':3,'Y':1},{'X':1,'Y':2},{'X':True,'Y':3},{'X':1}]:
            self.assertFalse(verify_assignment(self.ir,bad))

    def test_unsat_and_budget(self):
        self.ir['constraints'].append({'op':'lt','vars':['Y','X'],'value':0})
        self.assertEqual(solve_constraints(self.ir)['status'],'unsatisfiable')
        self.assertEqual(solve_constraints(self.ir,max_nodes=0)['status'],'budget_exhausted')

    def test_semantic_audit_detects_missing_rule(self):
        bad = copy.deepcopy(self.ir)
        bad['constraints'].pop()
        self.assertFalse(semantic_audit(self.ir,bad)['equivalent'])

    def test_semantic_audit_accepts_reordering(self):
        other = copy.deepcopy(self.ir)
        other['variables'].reverse()
        other['constraints'].reverse()
        self.assertTrue(semantic_audit(self.ir,other)['equivalent'])

    def test_invalid_references_rejected(self):
        self.ir['constraints'][0]['vars']=['X','UNKNOWN']
        with self.assertRaises(ValueError): validate(self.ir)

    def test_offset_and_distinct(self):
        self.ir['constraints']=[{'op':'offset','vars':['Y','X'],'value':1},
                                {'op':'all_different','vars':['X','Y'],'value':0},
                                {'op':'neq','vars':['X'],'value':2}]
        answer=solve_constraints(self.ir)['assignment']
        self.assertEqual(answer,{'X':3,'Y':2})


if __name__ == '__main__': unittest.main()
