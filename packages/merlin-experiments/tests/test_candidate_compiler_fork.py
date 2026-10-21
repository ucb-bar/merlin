"""Editable compiler sources stay candidate-owned and execute only in the box."""
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from test_compiler_libraries import policy
from merlin.targetgen.compiler_library import CompilerLibrarySelection, freeze_compiler_library, scoped_compiler_library
from merlin.targetgen.compiler_library_fork import candidate_sources, stage, REVIEW_PREFIX
from merlin_experiments.phase2 import compiler_libraries as L
from merlin_experiments.phase2.campaign import _verify_frozen_policy, CampaignGateError


def selected(tmp_path):
    root = tmp_path / 'baseline'
    leaves = {'merlin/__init__.py': '', 'merlin/transforms/__init__.py': '',
              'merlin/transforms/portable.py': 'def lower(x):\n return x + 1\n',
              'merlin/targetgen/__init__.py': '',
              'merlin/targetgen/address_space.py': 'FIXED = True\n'}
    sources=[]
    for relative, text in leaves.items():
        path=root/relative; path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
        module=relative.removesuffix('.py').replace('/','.')
        if module.endswith('.__init__'):module=module.removesuffix('.__init__')
        sources.append((relative,module))
    contract=freeze_compiler_library(root,review_id=REVIEW_PREFIX+'test',
        public_modules=('merlin.transforms.portable','merlin.targetgen.address_space'),sources=tuple(sources))
    selection=CompilerLibrarySelection(contract,root)
    package=tmp_path/'candidate';package.mkdir();stage(selection,package)
    return selection,package


def test_fork_rejects_ungranted_and_fixed_modules_links_and_changed_bytes(tmp_path):
    selection,package=selected(tmp_path)
    with scoped_compiler_library(selection):
        boxed=L.apply(policy(package))
        edited=package/'compiler_library/merlin/transforms/portable.py'
        edited.write_text('def lower(x):\n return x+2\n')
        with pytest.raises(CampaignGateError,match='changed'):_verify_frozen_policy(boxed)
    for relative in ('merlin/targetgen/address_space.py','merlin/targetgen/private_grader.py'):
        bad=package/'compiler_library'/relative;bad.parent.mkdir(parents=True,exist_ok=True);bad.write_text('pass\n')
        with pytest.raises(ValueError,match='ungranted'):candidate_sources(selection,package)
        bad.unlink()
    linked=package/'compiler_library/merlin/transforms/portable.py';linked.unlink();linked.symlink_to(selection.root/'merlin/transforms/portable.py')
    with pytest.raises(ValueError,match='links'):candidate_sources(selection,package)


def test_actual_edited_pass_is_used_by_isolated_emitter(tmp_path):
    binary=shutil.which('bwrap')
    if binary is None:pytest.skip('bwrap unavailable')
    selection,package=selected(tmp_path)
    (package/'compiler_library/merlin/transforms/portable.py').write_text('def lower(x):\n return x * 7\n')
    (package/'compiler.py').write_text('from merlin.transforms.portable import lower\nprint("emitted.constant="+str(lower(6)))\n')
    argv=[binary,'--unshare-all','--die-with-parent','--clearenv','--ro-bind','/usr','/usr']
    for name in ('/lib','/lib64'):
        if Path(name).exists():argv += ['--ro-bind',name,name]
    argv += ['--tmpfs','/tmp','--ro-bind',str(package),'/candidate']
    with scoped_compiler_library(selection):
        boxed=L.apply(policy(package,argv))
        _verify_frozen_policy(boxed)
        result=subprocess.run([*boxed.argv,'bash','-c',boxed.env_prefix+'exec /usr/bin/python3 -B /candidate/compiler.py'],capture_output=True,text=True,timeout=15)
        assert L.apply(boxed)==boxed
    if result.returncode and 'Operation not permitted' in result.stderr:pytest.skip('user namespaces unavailable')
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='emitted.constant=42'
    assert (selection.root/'merlin/transforms/portable.py').read_text().endswith('return x + 1\n')
    assert boxed.compiler_fork_sha256==candidate_sources(selection,package)[1]


def test_strict_component_policy_uses_the_same_fork(tmp_path):
    from merlin_experiments.phase1.component_package_execution import ComponentPackageExecutor
    from merlin_experiments.phase1.component_package_execution import _policy_environment
    selection,package=selected(tmp_path)
    edited=package/'compiler_library/merlin/transforms/portable.py'
    edited.write_text('def lower(x):\n return x*9\n')
    executor=ComponentPackageExecutor(package,None,(),tmp_path/'evidence')
    with scoped_compiler_library(selection):
        argv=executor._original_public_policy([],workspace=tmp_path/'scratch')
        assert str(edited) in argv
        assert _policy_environment(argv)['PYTHONPATH'].split(':')[0]==str(L.guest_root(selection))
    assert not (package/'compiler_library/merlin/targetgen/address_space.py').exists()


def test_fork_dependencies_never_grant_an_unlisted_grader(tmp_path):
    from merlin.targetgen.package_runtime import _py_imports_merlin
    selection,_=selected(tmp_path)
    assert _py_imports_merlin('from merlin.targetgen.private_grader import golden',
        compiler_library=selection.contract,compiler_dependency=True)=='merlin.targetgen.private_grader'
    assert _py_imports_merlin('from merlin.transforms import portable',
        compiler_library=selection.contract) is None
    assert _py_imports_merlin('import merlin.transforms.unlisted',
        compiler_library=selection.contract)=='merlin.transforms.unlisted'
