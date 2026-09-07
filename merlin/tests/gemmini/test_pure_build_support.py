"""Pure caller extraction: old emitted-byte pins, no candidate execution or simulator."""
import hashlib
import json
import subprocess
import sys

import pytest

from merlin.perf.storage_encoding import GroupedAxesStorage
from merlin.runtime.backends import base as bk
from merlin.common.paths import merlin_dir
from merlin.targetgen.contract.build_service import load_build_package


def fixture(dtype, explicit):
    tensors = {name: {'shape': [2, 3], 'dtype': dtype,
                       'role': 'input' if name == 'A' else 'output'} for name in ('A', 'Y')}
    cb = {'abi_version': '0.1', 'target': 'gemmini', 'commands': [], 'tensors': tensors,
          'kernel_abi': {'kind': 'whole_program', 'args': [
              {'tensor': 'A', 'access': 'read'}, {'tensor': 'Y', 'access': 'write'}],
              'outputs': ['Y']}}
    values = [-3, -2, -1, 0, 1, 2] if dtype.startswith('i') else [-1.5, -0.0, 0.25, 0.5, 1.75, -2.25]
    if explicit:
        encoding = GroupedAxesStorage((2, 3), dtype, ((0,), (1,)), (2, 3), (5, 1), 10)
        cb['params'] = {'storage_encodings': {name: encoding.to_dict() for name in tensors}}
        values = [values[:3], values[3:]]
    return cb, {'A': values}


# Captured from the original production renderer before the extraction. Includes
# warm/cold windows and integer/float physical storage rather than only headings.
ORIGINAL = {
    'i8:legacy:cold': 'cde8d4c780b01d0b165369b5fc34b07f6f17c1e3a70085b40bfe292ebbb1267f',
    'i8:legacy:warm': 'eef242bb1e28cac6052dbed6dade1b5d100acc9a1f9ab5ce39965762a4fb7a82',
    'i8:explicit:cold': '1e64bad095a5da65ae8be657eff47ba129e615567683b0650e0c77db53dcefa3',
    'i8:explicit:warm': '87d103dbccbe61bee813fe15f9ea14b751f3a02095ecbfa0d45f6722266d9cda',
    'i16:legacy:cold': '6512fd47f677dabf24ee316b289780ac063bf0c1d42df15057aa1d28c4c432a0',
    'i16:legacy:warm': '29328ddb37b941c4ee7786dd4ed4ac063a78fa8f213ea287448d4bf2f39f93be',
    'i16:explicit:cold': 'b53221520578f3e65c0eb01f4f1b0baeb2629a08702ae6e5c9932903ed08c8be',
    'i16:explicit:warm': 'ed44f4fea772078b8e617002e6cbcee3d8678c59454c9a9723c7bd25fb2d2e6a',
    'f32:legacy:cold': '5ff59151921ffd8a6885923fe9a341eddf7ce571d82c6630bbc3514ad48cadc5',
    'f32:legacy:warm': '71573b445e1d16b805bf3fede143ae28848a232effadf1649aadd4588f5cf4af',
    'f32:explicit:cold': 'e37ac1e6ff60a400ac8afb6e458f05195165daa1cdfd5e6188ac1912dc155809',
    'f32:explicit:warm': '2148b812e42f5ae6a87ec7030f14747062a7aaa35dee1bc47e9c420ea29707b7',
    'bf16:legacy:cold': '16af6c9428eca78e31673c85984b9f89e5b4da00365b7b6ad6a2f24b0424c041',
    'bf16:legacy:warm': '97ae261348c5ee5de27106db514d367f404c230c2cbf5f0ad3efdc11b34d44e5',
    'bf16:explicit:cold': 'b9b89c701728c2a5fcc01be306f69f7f8bfdf08c0df9c89ef088066a35694607',
    'bf16:explicit:warm': 'f6bec41f4057c0224a384e123b110f34d6dd45774f79921629762e9cdf79d8e7',
}


@pytest.mark.parametrize('key', list(ORIGINAL))
def test_default_backend_generated_bytes_are_unchanged(key, monkeypatch):
    dtype, explicit, state = key.split(':')
    monkeypatch.setenv('MERLIN_CACHE_STATE', state)
    monkeypatch.delenv('MERLIN_HW_COUNTERS', raising=False)
    cb, inputs = fixture(dtype, explicit == 'explicit')
    rendered = bk.get_backend('gemmini').render_harness(cb, target='gemmini', inputs=inputs)
    assert hashlib.sha256(rendered.encode()).hexdigest() == ORIGINAL[key]


def pure_package():
    return load_build_package(merlin_dir() / 'targets/gemmini/build_support/__init__.py')


def test_backend_aliases_and_target_source_hook_use_the_one_pure_implementation():
    backend = bk.get_backend('gemmini')
    pure = pure_package()
    assert backend.gemmini_codegen.CodegenError is pure.CodegenError
    assert backend.gemmini_codegen_mlir.Container is pure.Container
    assert backend.gemmini_codegen_mlir.container_for is pure.container_for
    assert backend.gemmini_codegen_mlir.container_words is pure.container_words
    assert tuple(backend.build_source_paths()) == pure.build_source_paths()
    assert {path.name for path in pure.build_source_paths()} == {
        '__init__.py', 'format.py', 'measurement.py', 'whole_program.py'}


def test_pure_default_is_compute_only_warm_one_without_backend_imports():
    cb, inputs = fixture('f32', True)
    request = json.dumps({'cb': cb, 'inputs': inputs,
        'package': str(merlin_dir() / 'targets/gemmini/build_support/__init__.py')})
    # Fresh process: a previously imported backend cannot disguise an implicit
    # dependency. Block both original backend entry and shared discovery path.
    program = '''
import builtins, json, sys
from pathlib import Path
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if (name.startswith('merlin.runtime.backends') or name.startswith('merlin._oot_backends')
            or name in ('merlin.runtime.reference', 'merlin.targetgen.capsule_dram')):
        raise AssertionError('masked dependency imported: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from merlin.targetgen.contract.build_service import load_build_package
from merlin.runtime import commandbuffer
def no_materialization(*args, **kwargs):
    raise AssertionError('explicit caller must not synthesize input values')
commandbuffer.materialize_inputs = no_materialization
request = json.loads(sys.stdin.read())
package = load_build_package(Path(request['package']))
text = package.render_whole_program(request['cb'], inputs=request['inputs'])
assert not any(name.startswith(('merlin.runtime.backends', 'merlin._oot_backends'))
               for name in sys.modules)
sys.stdout.write(text)
'''
    result = subprocess.run([sys.executable, '-c', program], input=request, text=True,
                            capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert hashlib.sha256(result.stdout.encode()).hexdigest() == ORIGINAL['f32:explicit:warm']
    assert result.stdout.count('gemmini_kernel((void*)T_A, (void*)T_Y);') == 2


def test_pure_legacy_layout_requires_explicit_host_callbacks():
    cb, inputs = fixture('i8', False)
    with pytest.raises(pure_package().CodegenError, match='explicit host layout helpers'):
        pure_package().render_whole_program(cb, inputs=inputs)


@pytest.mark.parametrize("dtype",["i8","i16","f32"])
def test_pure_legacy_exact_input_path_matches_original_bytes_without_materialization(dtype,monkeypatch):
    from merlin.runtime.tensor import Tensor
    from merlin.runtime import commandbuffer
    backend=bk.get_backend("gemmini")
    cb,values=fixture(dtype,False)
    inputs={"A":Tensor((2,3),values["A"],dtype)}
    monkeypatch.setattr(commandbuffer,"materialize_inputs",lambda *a,**k:pytest.fail("exact inputs cannot invoke materialization"))
    actual=pure_package().render_whole_program(cb,inputs=inputs,legacy_dim=backend.gemmini_codegen.DIM)
    assert hashlib.sha256(actual.encode()).hexdigest()==ORIGINAL[dtype+":legacy:warm"]


def test_measurement_final_assembly_does_not_discover_or_guess_counters():
    pure = pure_package()
    fragments = pure.assemble_measurement_fragments('  call();  ', requested='warm',
        include='header\n', prologue='reset\n', epilogue='snapshot\n')
    assert fragments['include'] == 'header\n'
    assert fragments['prologue'] == 'reset\n'
    assert fragments['epilogue'] == 'snapshot\n'
    assert fragments['warmup'].startswith('  call();\n')
    assert fragments['cache_state_observed'] is False
    with pytest.raises(pure.CodegenError, match='cache-state'):
        pure.assemble_measurement_fragments('call();', requested='observed_hot')
