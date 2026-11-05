"""Model-child routing controls; the synthetic profile fixture is not hardware qualification."""
import json
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin.targetgen.compiler_library import CompilerLibrarySelection, freeze_compiler_library, active_compiler_library
from merlin.targetgen.compiler_library_fork import REVIEW_PREFIX
from merlin_experiments.phase1.brokers import model_check_worker as worker
from merlin_experiments.phase2 import compiler_library_state as state
from merlin_experiments.phase2 import compiler_libraries as overlays
from merlin_experiments.phase2.campaign import PackageSandboxPolicy


def fixture(tmp_path):
    baseline = tmp_path / 'baseline'
    leaves = {'merlin/__init__.py': '', 'merlin/transforms/__init__.py': '',
              'merlin/transforms/portable.py': 'def lower(x):\n return x+1\n'}
    sources=[]
    for relative,text in leaves.items():
        p=baseline/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
        sources.append((relative,relative.removesuffix('.py').replace('/','.').removesuffix('.__init__')))
    c=freeze_compiler_library(baseline,review_id=REVIEW_PREFIX+'model-child',
                             public_modules=('merlin.transforms.portable',),sources=tuple(sources))
    selection=CompilerLibrarySelection(c,baseline)
    root=tmp_path/'run/compiler_library/python';binding=state.freeze_selection(selection,root)
    contract=root.parent/'contract.json';contract.write_bytes(__import__("merlin_experiments.phase1.compiler_library",fromlist=["_contract_bytes"])._contract_bytes(selection));contract.chmod(0o444)
    package=tmp_path/'candidate';override=package/'compiler_library/merlin/transforms/portable.py'
    override.parent.mkdir(parents=True);override.write_text('def lower(x):\n return x*7\n')
    (package/'manifest.yaml').write_text('commands: {}\n')
    (package/'compiler.py').write_text('from merlin.transforms.portable import lower\nprint(lower(6))\n')
    args=['--compiler-library-root',str(root),'--compiler-library-contract',str(contract),
          '--compiler-library-contract-sha256',c.sha256]
    return package,args


def test_model_worker_retains_explicit_owner_and_sparse_emitter(tmp_path,monkeypatch):
    from merlin.model_check import cli
    from merlin_experiments.phase1 import context
    from merlin_experiments.phase1.feedback import compiler_profile_runtime as profile
    binary=shutil.which('bwrap')
    if not binary:pytest.skip('bwrap unavailable')
    package,args=fixture(tmp_path);entered=[]
    monkeypatch.setattr(context,'resolve_context',lambda *a:SimpleNamespace(repo=tmp_path))
    @contextmanager
    def scope(manifest,**kwargs):
        assert active_compiler_library() is not None
        assert kwargs['submission']==package
        assert kwargs['work_root']==tmp_path/'work/compiler_execution'
        entered.append(True)
        yield
        entered.pop()
    monkeypatch.setattr(profile,'selfcheck_execution',scope)
    def invoke(argv):
        assert entered==[True]
        selection=active_compiler_library();assert selection is not None
        command=[binary,'--unshare-all','--die-with-parent','--clearenv','--ro-bind','/usr','/usr']
        for name in ('/lib','/lib64'):
            if Path(name).exists():command += ['--ro-bind',name,name]
        command += ['--ro-bind',str(package),'/candidate']
        policy=PackageSandboxPolicy(tuple(command),(),(),tmp_path/'scratch',package,SimpleNamespace(),env_prefix='')
        boxed=overlays.apply(policy)
        result=subprocess.run([*boxed.argv,'bash','-c',boxed.env_prefix+'exec /usr/bin/python3 -B /candidate/compiler.py'],capture_output=True,text=True,timeout=15)
        if result.returncode and 'Operation not permitted' in result.stderr:pytest.skip('user namespaces unavailable')
        assert result.returncode==0,result.stderr
        assert result.stdout.strip()=='42'
        assert 'merlin.transforms.portable' not in __import__('sys').modules
        return 0
    monkeypatch.setattr(cli,'main',invoke)
    assert worker.main([*args,'--compiler-bundle-manifest',str(tmp_path/'manifest.yaml'),
                        'check','--submission',str(package),'--work',str(tmp_path/'work')])==0
    assert active_compiler_library() is None and entered==[]


def test_model_worker_refuses_missing_profile_or_library(tmp_path,monkeypatch):
    from merlin.model_check import cli
    package,args=fixture(tmp_path)
    monkeypatch.setattr(cli,'main',lambda argv:pytest.fail('unowned compiler execution'))
    model=['check','--submission',str(package),'--work',str(tmp_path/'work')]
    with pytest.raises(ValueError,match='original profile and library owner'):worker.main([*args,*model])
    with pytest.raises(ValueError,match='original profile and library owner'):worker.main(model)


def test_model_cache_binds_selected_compiler_owner(tmp_path):
    from merlin.model_check.cache import compiler_owner_identity
    from merlin.targetgen.compiler_library import scoped_compiler_library
    from merlin_experiments.phase2.compiler_library_state import selection_from_arguments
    package,args=fixture(tmp_path)
    selection=selection_from_arguments(SimpleNamespace(compiler_library_root=Path(args[1]),
        compiler_library_contract=Path(args[3]),compiler_library_contract_sha256=args[5]))
    unbound=compiler_owner_identity()
    with scoped_compiler_library(selection):
        bound=compiler_owner_identity()
        assert bound['library']==selection.record() and bound!=unbound
    assert compiler_owner_identity()==unbound
