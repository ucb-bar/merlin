"""Fresh one-pass weight discovery preserves rglob's selected-root semantics."""
from pathlib import Path
from types import SimpleNamespace

import importlib

A = importlib.import_module("merlin.targetgen.sandbox.answer_surfaces")


def test_weight_walk_matches_prior_suffixes_and_observes_new_members(tmp_path, monkeypatch):
    root=tmp_path/'corpus'; child=root/'model'; child.mkdir(parents=True)
    (child/'a.safetensors').write_bytes(b'weights')
    (child/'a.safetensors.manifest.json').write_text('{}')
    external=tmp_path/'outside';external.mkdir()
    (external/'hidden.safetensors').write_bytes(b'other')
    (child/'dir-alias').symlink_to(external, target_is_directory=True)
    (child/'leaf.safetensors').symlink_to(external/'hidden.safetensors')
    monkeypatch.setattr(A,'contract_resource_roots',lambda *_: [root])
    te=SimpleNamespace(capsule_corpus=root/'isa')
    def prior():
        return sorted({p for pattern in ('*.safetensors','*.safetensors.manifest.json')
                       for p in root.rglob(pattern) if p.is_file()})
    assert A.weight_files(te)==prior()
    (child/'late.safetensors').write_bytes(b'new')
    assert A.weight_files(te)==prior()
    assert child/'dir-alias/hidden.safetensors' not in A.weight_files(te)
    monkeypatch.setattr(A,'contract_resource_roots',lambda *_: [child/'dir-alias'])
    selected=SimpleNamespace(capsule_corpus=None)
    assert A.weight_files(selected)==[child/'dir-alias/hidden.safetensors']
