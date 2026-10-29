from types import SimpleNamespace

import pytest

from merlin.targetgen import capsule_privacy
from merlin.targetgen.sandbox import bwrap as B


def fixture(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    destination = repo / 'corpus'
    source = tmp_path / 'snapshot'
    cap = source / 'cases/one'
    cap.mkdir(parents=True)
    (cap / 'capsule.yaml').write_text('name: one\n')
    (cap / 'golden.yaml').write_text('private: true\n')
    (cap / 'expected_instruction_coverage.yaml').write_text('{}\n')
    monkeypatch.setattr(B, '_snapshot_grants', lambda *args: ({}, [('grant', destination, source)]))
    # The same tree has both classifications. Corpus roots overlap twice.
    monkeypatch.setattr(B, 'contract_resource_roots', lambda _repo, kind:
                        (destination,) if kind in ('capsules', 'examples') else ())
    target = SimpleNamespace(capsule_corpus=destination / 'cases', corpus_siblings=lambda: ())
    return repo, source, cap, target


def test_overlap_parses_once_and_retains_distinct_classifications(tmp_path, monkeypatch):
    repo, source, cap, target = fixture(tmp_path, monkeypatch)
    original = capsule_privacy.capsule_answer_members
    visits = []
    def counted(path, **kwargs):
        visits.append(path)
        return original(path, **kwargs)
    monkeypatch.setattr(capsule_privacy, 'capsule_answer_members', counted)
    surfaces = B._frozen_answer_surfaces(tmp_path, {}, target, [], repo)
    by_path = {s.path: s.origin for s in surfaces}
    assert visits == [cap]
    assert by_path[cap / 'golden.yaml'] == 'golden'
    # The example classification runs independently after corpus classification.
    assert by_path[cap / 'expected_instruction_coverage.yaml'] == 'example'


def test_next_policy_rereads_changed_metadata_and_refuses_malformed(tmp_path, monkeypatch):
    repo, source, cap, target = fixture(tmp_path, monkeypatch)
    B._frozen_answer_surfaces(tmp_path, {}, target, [], repo)
    (cap / 'capsule.yaml').write_text('reference_closure: invalid\n')
    with pytest.raises(ValueError, match='closure is malformed'):
        B._frozen_answer_surfaces(tmp_path, {}, target, [], repo)
