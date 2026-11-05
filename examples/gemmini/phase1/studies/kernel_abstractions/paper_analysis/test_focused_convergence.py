import unittest

from plot_focused_convergence import first_source_reaching, sources_and_curve


def row(t, source, case, status):
    return dict(record_type='grade_case', elapsed_s=str(t), publication_elapsed_s=str(t),
                source_sha256=source, case_id=case, l2_status=status,
                record_id=f'{t}:{source}:{case}')


class SourceCoherence(unittest.TestCase):
    def test_different_sources_do_not_supply_a_joint_compiler(self):
        rows = [row(1, 'a', 'x', 'pass'), row(2, 'b', 'y', 'pass')]
        _, curve, _ = sources_and_curve(rows)
        self.assertEqual(curve[-1][1], 1)
        self.assertIsNone(first_source_reaching(rows, {'x', 'y'}))

    def test_same_source_can_accumulate_immutable_grading_batches(self):
        rows = [row(1, 'a', 'x', 'pass'), row(2, 'a', 'y', 'pass')]
        milestone = first_source_reaching(rows, {'x', 'y'})
        self.assertEqual(milestone['source_sha256'], 'a')
        self.assertEqual(milestone['at_s'], 2)

    def test_unobserved_status_does_not_retract_a_pass(self):
        rows = [row(1, 'a', 'x', 'pass'), row(2, 'a', 'x', 'unavailable')]
        _, curve, _ = sources_and_curve(rows)
        self.assertEqual(curve[-1][1], 1)

    def test_explicit_failure_retracts_prior_pass(self):
        rows = [row(1, 'a', 'x', 'pass'), row(2, 'a', 'x', 'fail')]
        _, curve, _ = sources_and_curve(rows)
        self.assertEqual(curve[-1][1], 0)

    def test_comparison_curve_uses_the_identical_fixed_case_set(self):
        rows = [row(1, 'a', 'x', 'pass'), row(2, 'a', 'outside_goal', 'pass')]
        _, curve, _ = sources_and_curve(rows, case_ids={'x'})
        self.assertEqual(curve[-1][1], 1)


if __name__ == '__main__':
    unittest.main()
