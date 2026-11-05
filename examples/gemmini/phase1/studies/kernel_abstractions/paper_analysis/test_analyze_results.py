import unittest

from analyze_results import distinct_attempts, repairs, time_of, token_points


class EventSemantics(unittest.TestCase):
    def test_availability_requires_publication(self):
        self.assertEqual(time_of(dict(elapsed_s='60', publication_elapsed_s='120')), 120)

    def test_republication_is_not_another_attempt(self):
        row = dict(record_type='selfcheck_case', attempt_id='request', source_sha256='s',
                   case_id='x', elapsed_s='10', publication_elapsed_s='10', l2_status='fail')
        updated = dict(row, elapsed_s='20', l2_status='pass')
        selected = distinct_attempts([row, updated])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['l2_status'], 'pass')

    def test_existing_metric_aggregate_does_not_duplicate_tokens(self):
        rows = [dict(record_type='provider_usage', elapsed_s='1', tokens_total='100', tokens_fresh_input='10', tokens_output='5'),
                dict(record_type='small_metric', elapsed_s='2', tokens_total='100', tokens_fresh_input='10', tokens_output='5')]
        self.assertEqual(token_points(rows), ([1.0], [100.0], [15.0]))

    def test_open_repair_is_censored_and_closed_repair_keeps_source_change(self):
        failed = dict(case_id='x', elapsed_s='10', l2_status='fail', source_sha256='old', failure_plane='spike')
        passed = dict(case_id='x', elapsed_s='30', l2_status='pass', source_sha256='new')
        other = dict(case_id='y', elapsed_s='40', l2_status='fail', source_sha256='new')
        out = repairs([failed, passed, other])
        self.assertEqual(out['open_episodes_censored'], 1)
        self.assertEqual(out['closed_episodes'][0]['elapsed_s'], 20)
        self.assertTrue(out['closed_episodes'][0]['changed_source'])


if __name__ == '__main__':
    unittest.main()
