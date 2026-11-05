"""A real sealed original worker return can attest one completed preparation."""

import json
import os
import subprocess
import sys
from copy import deepcopy

import pytest
from merlin_experiments.phase1.feedback import compiler_profile_child as CHILD
from merlin_experiments.phase1.feedback import model_storage_completion as M
from merlin_experiments.phase1.feedback import snapshot_execution as S
from test_compiler_profile_child import _scope, original  # noqa: F401 -- original selected-owner fixture

from merlin.llvmlower.completed_host_storage import digest
from merlin.llvmlower.host_graph_policy import read_host_graph_policy
from merlin.targetgen.source_composition_preferences import require_host_preparation

SCRIPT = r'''
import importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
from merlin.llvmlower.borrowed_arena import retain_borrowed_lifetime
from merlin.llvmlower.host_storage_policy import prepare,recheck
from merlin.llvmlower.lower import lower_model
from merlin.targetgen.source_composition_preferences import require_host_preparation
from merlin_experiments.phase1.feedback import compiler_profile_child as CHILD
from merlin_experiments.phase1.feedback import model_storage_completion as M
from merlin_experiments.phase1.feedback import snapshot_execution as S,snapshot_model_child as SM
spec=Path(sys.argv[2]);raw=spec.read_bytes()
restore=SM.restore if S.selected() else CHILD.restore_model_child
with restore(int(sys.argv[1]),spec_path=spec,spec_bytes=raw) as owner:
 d=spec.parent/'storage-probe';d.mkdir()
 helper=Path(sys.argv[3])/'merlin/tests/ir/test_borrowed_arena_native.py'
 loader=importlib.util.spec_from_file_location('fixture',helper)
 fixture=importlib.util.module_from_spec(loader);loader.loader.exec_module(fixture)
 s=fixture.source(2,3,4)
 selected,_=fixture.selection(d,s)
 result=lower_model(s,d/'lower',targets=(),borrowed_buffer_calls=selected,
                    static_arena=False,hoist_static_allocs=False)
 policy=fixture.storage_policy(d)
 prepare(result.ll_path,policy,d/'storage',SimpleNamespace(bind_preparation=lambda *args:None),
         borrowed_lifetime=result.borrowed_arena_lifetime)
 artifacts=recheck(d/'storage',policy,borrowed_lifetime=result.borrowed_arena_lifetime)
 receipt=retain_borrowed_lifetime(dict(inputs=dict(device=dict(intermediate_storage=policy.record())),
                                     output=dict(intermediate_storage=artifacts)),result.borrowed_arena_lifetime)
 require_host_preparation(receipt,dict(intermediate_storage=policy))
 returned={SM.FIELD if S.selected() else 'original_compiler_child':owner,
           'owned_host_native_execution':dict(build=receipt)}
 M.export_return(returned)
print(json.dumps(returned))
'''


@pytest.mark.parametrize('snapshot', [False, True])
def test_actual_sealed_child_return_rejoins_exact_storage_without_exporting_effects(original, monkeypatch, snapshot):  # noqa: F811
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    if snapshot:
        monkeypatch.setenv(S.ENV, S.MODE)
    else:
        monkeypatch.delenv(S.ENV, raising=False)
    scope = (S.execution(original.owner, target_experiment=original.target, candidate=original.candidate,
                         work_root=original.private/'external', sandbox_inputs=original.inputs)
             if snapshot else _scope(original))
    with scope:
        with CHILD.prepare_model_child(spec_path=original.spec, environment=dict(os.environ),
                                       package_dir=original.candidate) as transport:
            command = [sys.executable, '-c', SCRIPT, str(transport.fd), str(original.spec), str(repo)]
            process = subprocess.Popen(command, pass_fds=transport.pass_fds, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, env=dict(os.environ))
            transport.launched(process, command)
            stdout, stderr = process.communicate(timeout=30)
            assert process.returncode == 0, stderr
            returned = json.loads(stdout.splitlines()[-1])
            saved = deepcopy(returned)
            assert M.FIELD in returned
            assert transport.document['storage_completion_key'] not in json.dumps(returned)
            with pytest.raises(ValueError, match='actual successful child'):
                transport.verify_result(deepcopy(saved))
            changed = deepcopy(saved)
            changed['owned_host_native_execution']['build']['changed'] = True
            changed[M.FIELD]['preparation']['build_sha256'] = digest(changed['owned_host_native_execution']['build'])
            changed[M.FIELD]['result_sha256'] = digest({k: v for k, v in changed.items() if k != M.FIELD})
            with pytest.raises(ValueError, match='original trusted worker completion'):
                transport.verify_result(changed, completed_process=process)
            directory = original.spec.parent/'storage-probe'
            for path in (directory/'implementation.c', directory/'storage/selected.ll'):
                before = path.read_bytes()
                path.write_bytes(before+b'\n')
                try:
                    with pytest.raises(ValueError, match='source or product changed'):
                        transport.verify_result(deepcopy(saved), completed_process=process)
                finally:
                    path.write_bytes(before)
            arguments = process.args
            process.args = [*arguments, 'changed-worker']
            with pytest.raises(ValueError, match='actual successful child'):
                transport.verify_result(deepcopy(saved), completed_process=process)
            process.args = arguments
            # Selected original host bytes are independently reopened too.
            before = original.frozen.read_bytes()
            mode = original.frozen.stat().st_mode & 0o777
            original.frozen.chmod(0o600)
            original.frozen.write_bytes(before+b'\n')
            try:
                with pytest.raises(Exception, match='changed|digest|snapshot|integrity'):
                    transport.verify_result(deepcopy(saved), completed_process=process)
            finally:
                original.frozen.write_bytes(before)
                original.frozen.chmod(mode)
            observed = transport.verify_result(returned, completed_process=process)
            assert M.FIELD not in observed
            policy = read_host_graph_policy(directory/'software.yaml', target='neutral', entry='forward')
            options = dict(intermediate_storage=policy)
            receipt = observed['owned_host_native_execution']['build']
            require_host_preparation(receipt, options)
            for copied in (dict(receipt), deepcopy(receipt), json.loads(json.dumps(receipt))):
                with pytest.raises(ValueError, match='changed'):
                    require_host_preparation(copied, options)
            with pytest.raises(ValueError, match='actual successful child'):
                transport.verify_result(saved, completed_process=process)
        # The observation is a completed operation; it needs no deleted worker
        # FD or cross-process effects grant for its exact retained products.
        require_host_preparation(receipt, options)


def test_failed_actual_worker_cannot_restore_a_saved_completion(original):  # noqa: F811
    with _scope(original):
        with CHILD.prepare_model_child(spec_path=original.spec, environment=dict(os.environ),
                                       package_dir=original.candidate) as transport:
            command = [sys.executable, '-c', 'raise SystemExit(7)']
            process = subprocess.Popen(command)
            transport.launched(process, command)
            process.wait(timeout=10)
            result = dict(original_compiler_child=CHILD._receipt(transport.document))
            result[M.FIELD] = {}
            with pytest.raises(ValueError, match='actual successful child'):
                transport.verify_result(result, completed_process=process)
