from pathlib import Path
from types import SimpleNamespace

from merlin.targetgen.sandbox import bwrap as bw
from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface


def test_frozen_projection_preserves_duplicate_order_and_component_boundaries(monkeypatch):
    destination = Path('/repo/lib')
    source = Path('/frozen/lib')
    live = [AnswerSurface('outside', Path('/repo/library/x'), 'file', 'hidden'),
            AnswerSurface('first', destination/'x', 'file', 'golden'),
            AnswerSurface('root', destination, 'dir', 'hidden'),
            AnswerSurface('last', destination/'x', 'file', 'weight')]
    seen = []
    monkeypatch.setattr(bw, '_snapshot_grants', lambda *a: ({}, [('lib', destination, source)]))
    monkeypatch.setattr(bw, 'contract_resource_roots', lambda *a: ())
    def kind(path):
        seen.append(path)
        return 'dir' if path == source else 'file'
    monkeypatch.setattr(bw, 'path_kind', kind)
    target = SimpleNamespace(capsule_corpus=None, corpus_siblings=lambda: ())
    result = bw._frozen_answer_surfaces(Path('/ws'), {}, target, live, Path('/repo'))
    assert seen == [source/'x', source, source/'x']
    assert [(row.path, row.origin) for row in result] == [(source/'x','weight'), (source,'hidden')]


def test_index_matches_original_ordered_membership_including_duplicates():
    paths = [Path('/a/b'), Path('/a/bc'), Path('/a/b/x'), Path('/a/b'), Path('/z')]
    index = bw._MountPathIndex((path,) for path in paths)
    for root in [Path('/'), Path('/a'), Path('/a/b'), Path('/a/b/x'), Path('/missing')]:
        assert sorted(index.descendants(root)) == [i for i,p in enumerate(paths) if p.is_relative_to(root)]
