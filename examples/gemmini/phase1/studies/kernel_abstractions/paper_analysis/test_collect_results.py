"""Measurement boundary regression tests; no author/model/simulator calls."""
import unittest
from pathlib import Path

from collect_results import Collector, explicit_failure


class MeasurementBoundaries(unittest.TestCase):
    def setUp(self):
        self.c = Collector(Path('/unused'), Path('/unused'), cutoff=1000)
        self.meta = dict(run_id='r', experiment='el1', cohort='test', framework_revision='f',
                         seed_origin='empty_backend', author_budget_s=100, started_epoch=100,
                         author_started_at='1970-01-01T00:01:40+00:00', drain_s=10,
                         case_suites={}, public_roster=['SY_refuse_requant', 'x'])

    def cases(self, rows, **extra):
        self.c.cases(self.meta, dict(per_capsule=rows, graded_source_sha256='s',
                                    count_scope='immutable_snapshot', **extra),
                     Path('/evidence'), 'grade', publication=200, grade_id='g')
        return self.c.rows

    def test_expected_refusal_is_not_a_numerical_execution(self):
        records = self.cases([dict(capsule='SY_refuse_requant', status='pass', **{'pass': True}, tiers={})])
        self.assertIsNone(records[0]['numeric_status'])
        self.assertEqual(records[-1]['n_refused'], 1)
        self.assertEqual(records[-1]['n_l2_passed'], 0)

    def test_late_publication_is_excluded_even_with_early_origin(self):
        self.c.cases(self.meta, dict(per_capsule=[dict(capsule='x', graded={'graded_at': 200})]),
                     Path('/evidence'), 'grade', publication=1001, grade_id='g')
        self.assertEqual(self.c.rows, [])

    def test_cross_source_rows_prevent_coherent_score(self):
        rows = [dict(capsule='x', graded={'graded_source_sha256': 'another'}, tiers={'L2': 'pass'})]
        self.assertFalse(self.cases(rows)[-1]['coherent_snapshot'])

    def test_unavailable_is_pending_and_gated_is_counted(self):
        records = self.cases([dict(capsule='x', status='gated', tiers={}),
                              dict(capsule='y', status='incomplete', tiers={'L2': 'pass', 'L3': 'unavailable'})])
        self.assertEqual(records[-1]['n_failed'], 0)
        self.assertEqual(records[-1]['n_pending'], 2)
        self.assertEqual(records[-1]['n_gated'], 1)
        self.assertEqual(records[-1]['n_l2_passed'], 1)

    def test_numeric_pass_does_not_erase_failed_execution(self):
        r = dict(capsule='x', numeric_status='pass', tiers={'L2': 'fail'})
        records = self.cases([r])
        self.assertTrue(explicit_failure(r))
        self.assertEqual(records[-1]['n_l2_passed'], 0)
        self.assertEqual(records[-1]['n_failed'], 1)

    def test_violated_refusal_does_not_count_as_successful_refusal(self):
        r = dict(capsule='SY_refuse_requant', status='fail',
                 failure_plane='expected_refusal', failure_category='MUST_REFUSE_VIOLATED')
        records = self.cases([r])
        self.assertEqual(records[-1]['n_refused'], 0)
        self.assertEqual(records[-1]['n_failed'], 1)


if __name__ == '__main__':
    unittest.main()
