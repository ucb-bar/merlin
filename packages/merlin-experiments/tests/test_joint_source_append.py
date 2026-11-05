"""Reviewed additions never grant removal or mutation of existing obligations."""
import copy

import pytest

from merlin_experiments.phase1 import joint_source_append as A, joint_source_revision as R


def test_append_normalizes_only_reviewed_new_members(monkeypatch):
    initial = {'members': {'existing': {'logical_read_bytes': 8}}, 'runtime': 'same'}
    revision = {'schema': 'merlin.source_member_append_revision.v1', 'review_id': 'reviewed', 'evidence': {}}
    current = {**copy.deepcopy(initial), 'source_binding_revisions': [revision]}
    current['members']['new'] = {'logical_read_bytes': 16}
    monkeypatch.setattr(R.PERF, '_read', lambda pin: {'reviewed': True})
    monkeypatch.setattr(A, 'verified_members', lambda evidence: {'new': {'logical_read_bytes': 16}})
    assert R.normalized(initial, current) == initial
    for changed in ('existing', 'new', 'unknown'):
        bad = copy.deepcopy(current)
        bad['members'][changed] = {'logical_read_bytes': 99}
        with pytest.raises(ValueError, match='unreviewed member'):
            R.normalized(initial, bad)
    monkeypatch.setattr(A, 'verified_members', lambda evidence: {'existing': {'logical_read_bytes': 8}})
    with pytest.raises(ValueError, match='replace existing'):
        R.normalized(initial, current)


def test_unbounded_or_ambiguous_review_refuses():
    for rows in ([], [{}]*65, {}):
        with pytest.raises(ValueError, match='bounded nonempty'):
            A.verified_members({'schema': 'merlin.source_member_append_review.v1', 'members': rows})
    with pytest.raises(ValueError, match='explicit member review'):
        A.verified_members({'schema': 'merlin.source_member_append_review.v1', 'members': [{}], 'qualifies': True})
