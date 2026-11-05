"""Background gate retains original library and explicit profile, no target run."""
import contextvars
import threading
from pathlib import Path

import pytest

from merlin.targetgen.compiler_library import CompilerLibrarySelection, freeze_compiler_library, scoped_compiler_library, active_compiler_library
from merlin_experiments.phase1 import compiler_library as owner
from merlin_experiments.phase1.feedback import visible_model_gate as G
from merlin.common.tree_hash import hash_tree


@pytest.fixture
def selected(tmp_path,monkeypatch):
    monkeypatch.delenv('MERLIN_JOINT_CAMPAIGN_STATE',raising=False)
    root=tmp_path/'library';(root/'merlin').mkdir(parents=True)
    (root/'merlin/__init__.py').write_text('')
    (root/'merlin/portable.py').write_text('def lower(x): return x\n')
    contract=freeze_compiler_library(root,review_id='model-gate-test',public_modules=('merlin.portable',),
        sources=(('merlin/__init__.py','merlin'),('merlin/portable.py','merlin.portable')))
    selected=CompilerLibrarySelection(contract,root)
    path=tmp_path/'contract.json';path.write_bytes(owner._contract_bytes(selected));path.chmod(0o444)
    ws=tmp_path/'ws';(ws/'submission/compiler_library/merlin').mkdir(parents=True)
    (ws/'submission/manifest.yaml').write_text('commands: {}\n')
    (ws/'submission/compiler_library/merlin/portable.py').write_text('def lower(x): return x+1\n')
    run=tmp_path/'run';run.mkdir()
    return selected,path,ws,run


def test_background_gate_transfers_original_owner_and_explicit_profile(selected,monkeypatch):
    library,contract,ws,run=selected
    context=object();manifest=run/'input_bundle_manifest.yaml'
    seen=[];entered=threading.Event();unrelated=contextvars.ContextVar('unrelated',default=None)
    unrelated.set('must not transfer')
    def execute(_ws, request, **kwargs):
        assert active_compiler_library() is library
        assert owner.child_arguments()[1]==str(library.root)
        assert kwargs['context'] is context and kwargs['compiler_bundle_manifest']==manifest
        assert unrelated.get() is None
        seen.append(request);entered.set()
        return {'status':'pass','source_sha256':hash_tree(kwargs['snapshot_source'])['sha256'],
            'report':{'model_metrics':{'worst':{}},'programs':[{'program':p,'census':{'denominator':1,'device':1}} for p in G.PROGRAMS[request['model']]]}}
    monkeypatch.setattr(G,'execute_visible',execute)
    with scoped_compiler_library(library):
        token=owner._INITIAL.set((library,contract))
        try:
            handle=G.start(ws,run,context=context,compiler_bundle_manifest=manifest,interval_s=30)
            try: assert entered.wait(2)
            finally: G.stop_gate(handle)
            assert seen
            assert active_compiler_library() is library
        finally:owner._INITIAL.reset(token)
    assert active_compiler_library() is None


def test_background_owner_final_mutation_refuses_at_join(selected,monkeypatch):
    library,contract,ws,run=selected
    entered=threading.Event();release=threading.Event()
    def execute(*args,**kwargs):
        entered.set();assert release.wait(2)
        return {'status':'error','source_sha256':hash_tree(kwargs['snapshot_source'])['sha256']}
    monkeypatch.setattr(G,'execute_visible',execute)
    original=contract.read_bytes()
    with scoped_compiler_library(library):
        token=owner._INITIAL.set((library,contract))
        try:
            handle=G.start(ws,run,context=object(),compiler_bundle_manifest=run/'input_bundle_manifest.yaml')
            assert entered.wait(2)
            contract.chmod(0o644);contract.write_text('{}');handle[1].set();release.set()
            with pytest.raises((RuntimeError,ValueError),match='contract changed|background owner'):
                G.stop_gate(handle)
        finally:
            contract.write_bytes(original);contract.chmod(0o444);owner._INITIAL.reset(token)
