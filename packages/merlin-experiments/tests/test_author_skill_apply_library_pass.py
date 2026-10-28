"""Actual declared emission must consume the shared registered pass application."""
from __future__ import annotations

import json
import sys

import pytest
from author_skill_fixtures import load_script

from merlin.common.paths import checkout_root
from merlin.transforms.certificates import parse_ir

ADAPTER = '''from merlin.transforms.certificates import ir_text, parse_ir
from merlin.transforms.pipeline import PipelineOptions, apply_common_pipeline
from merlin.transforms.pipeline_policy import (PipelinePolicy, PipelineVariant,
    PolicyFacts, RegisteredPass)
from merlin.transforms.pipeline_state import PassSpec, PipelineSession

def source(case, *, package):
    return package / 'source.mlir'

def prepare(source, *, package, mechanism):
    def fold(s):
        s.module = apply_common_pipeline(s.module, PipelineOptions(64, reuse_buffers=False),
                                         plan_only=True).module
        return {}
    session = PipelineSession(parse_ir(source.read_text()))
    registry = {
        'fold': RegisteredPass(PassSpec('fold', 0), fold),
        'emit': RegisteredPass(PassSpec('emit', 1, produces=('product',), mutates_ir=False),
                               lambda s: {'product': ir_text(s.module)}),
    }
    paths = {None: ('emit',), 'fold': ('fold', 'emit'), 'noop': ('emit',)}
    policy = PipelinePolicy((PipelineVariant('selected', paths[mechanism]),), 'selected')
    return dict(session=session, facts=PolicyFacts(session.stamp()), registry=registry,
                policy=policy, output='product')
'''

COMPILER = '''import json, sys
from pathlib import Path
from library_pass_adapter import prepare
from merlin.transforms.pipeline_policy import execute_policy
package = Path.cwd()
selection = package / 'LIBRARY_PASS_SELECTION.json'
mechanism = json.loads(selection.read_text())['mechanism'] if selection.exists() else None
binding = prepare(Path(sys.argv[1]), package=package, mechanism=mechanism)
sys.stdout.write(execute_policy(**binding).artifact)
'''


@pytest.fixture
def package(tmp_path):
    base = tmp_path / 'submission'
    (base / 'tools').mkdir(parents=True)
    (base / 'tools/library_pass_adapter.py').write_text(ADAPTER)
    (base / 'tools/compiler.py').write_text(COMPILER)
    (base / 'manifest.yaml').write_text(json.dumps({'commands': {'lower_target_to_llvm': {
        'argv': [sys.executable, 'tools/compiler.py', '{input_mlir}']}}}))
    (base / 'source.mlir').write_text('''builtin.module { func.func @f() -> i32 {
      %a = arith.constant 3 : i32
      %b = arith.constant 4 : i32
      %sum = arith.addi %a, %b : i32
      func.return %sum : i32
    }}''')
    return base


def invoke(package, tmp_path, mechanism='fold', *, cli=None, **extra):
    cli = cli or load_script('ab-library-pass', 'apply_library_pass')
    receipt = tmp_path / 'application.json'
    args = ['--submission', str(package), '--variant', str(tmp_path / 'variants/changed'),
            '--library', str(checkout_root() / 'src'), '--mechanism', mechanism,
            '--case', 'source-locator', '--receipt', str(receipt)]
    for key, value in extra.items():
        args.extend(['--' + key, str(value)])
    code = cli.main(args)
    return code, json.loads(receipt.read_text())


def evaluate(path):
    from xdsl.interpreter import Interpreter
    from xdsl.interpreters.arith import ArithFunctions
    from xdsl.interpreters.func import FuncFunctions

    interpreter = Interpreter(parse_ir(path.read_text()))
    interpreter.register_implementations(ArithFunctions())
    interpreter.register_implementations(FuncFunctions())
    return interpreter.call_op('f', ())


def test_registered_pass_changes_actual_emission_and_preserves_neutral_numerics(package, tmp_path):
    (package / 'build').mkdir()
    (package / 'build/cache').write_bytes(b'not source')
    original = (package / 'source.mlir').read_bytes()
    code, record = invoke(package, tmp_path)
    assert code == 0 and record['status'] == 'APPLIED_EMITTED_CHANGE'
    assert record['actual_emitted_products_differ']
    base = tmp_path / 'application-products/baseline/stdout'
    variant = tmp_path / 'application-products/variant/stdout'
    assert 'arith.addi' in base.read_text() and 'arith.addi' not in variant.read_text()
    assert evaluate(base) == evaluate(variant) == (7,)
    assert [p['name'] for p in record['variant_pipeline']['passes']] == ['fold', 'emit']
    assert record['numerical_status'] == 'NOT_CHECKED' and record['timing_status'] == 'UNKNOWN'
    assert not (tmp_path / 'variants/changed/build').exists()
    assert (tmp_path / 'variants/changed/compiler_library').is_symlink()
    assert record['selected_library']['reused']
    assert not (package / 'LIBRARY_PASS_SELECTION.json').exists()
    assert (package / 'source.mlir').read_bytes() == original


def test_missing_hook_requests_construction_without_a_variant(package, tmp_path):
    (package / 'tools/library_pass_adapter.py').unlink()
    code, record = invoke(package, tmp_path)
    assert code == 2 and record['status'] == 'CONSTRUCTION_REQUIRED'
    assert 'Implement author adapter' in record['detail']
    assert not (tmp_path / 'variants/changed').exists()


def test_manifest_tool_placeholder_uses_the_copied_package_entrypoint(package, tmp_path):
    manifest = {'entrypoints': {'tool': 'tools/compiler.py'}, 'commands': {'lower_target_to_llvm': {
        'argv': [sys.executable, '{tool}', '{input_mlir}']}}}
    (package / 'manifest.yaml').write_text(json.dumps(manifest))
    code, record = invoke(package, tmp_path)
    assert code == 0 and record['actual_emitted_products_differ']
    baseline = record['baseline_emission']['argv']
    variant = record['variant_emission']['argv']
    assert baseline[1] == str(package / 'tools/compiler.py')
    assert variant[1] == str(tmp_path / 'variants/changed/tools/compiler.py')
    assert evaluate(tmp_path / 'application-products/variant/stdout') == (7,)


@pytest.mark.parametrize('mechanism,status,detail', [
    ('noop', 'REFUSED', 'UNCHANGED'),
    ('unknown', 'CONSTRUCTION_REQUIRED', 'no declared mapping'),
])
def test_noop_or_missing_mapping_cannot_proceed_to_evaluation(package, tmp_path, mechanism, status, detail):
    from merlin_experiments.phase1.feedback.idle_tuning import _variant_names

    code, record = invoke(package, tmp_path, mechanism)
    assert code != 0 and record['status'] == status and detail in record['detail']
    assert (tmp_path / 'variants/changed/manifest.yaml').exists()
    assert _variant_names(tmp_path, 8) == []
    assert (tmp_path / 'variants/.library-pass-pending-changed').is_file()


def test_manifest_ignoring_selection_is_refused(package, tmp_path):
    path = package / 'tools/compiler.py'
    path.write_text(COMPILER.replace('mechanism=mechanism', 'mechanism=None'))
    code, record = invoke(package, tmp_path)
    assert code == 1 and 'DISCONNECTED_ADAPTER' in record['detail']


def test_safe_link_is_materialized_as_normal_variant_source(package, tmp_path):
    (package / 'same-source.mlir').symlink_to('source.mlir')
    code, _ = invoke(package, tmp_path)
    assert code == 0
    copied = tmp_path / 'variants/changed/same-source.mlir'
    assert copied.is_file() and not copied.is_symlink()
    assert copied.read_bytes() == (package / 'source.mlir').read_bytes()


def test_unsafe_link_is_refused_before_copy(package, tmp_path):
    outside = tmp_path / 'outside'
    outside.write_text('not selected source')
    (package / 'escape').symlink_to(outside)
    code, record = invoke(package, tmp_path)
    assert code == 1 and 'unsafe source link' in record['detail']
    assert not (tmp_path / 'variants/changed').exists()


def test_idle_discovery_waits_for_success_receipt_after_actual_emission(package, tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback.idle_tuning import _variant_names

    cli = load_script('ab-library-pass', 'apply_library_pass')
    original_emit = cli.emit
    original_copy = cli.shutil.copy2
    seen = []
    copied = []

    def inspect_copy(source, destination):
        assert (tmp_path / 'variants/.library-pass-pending-changed').is_file()
        assert _variant_names(tmp_path, 8) == []
        copied.append(True)
        return original_copy(source, destination)

    def inspect_pending(*args, **kwargs):
        assert (tmp_path / 'variants/changed/manifest.yaml').exists()
        assert (tmp_path / 'variants/.library-pass-pending-changed').is_file()
        assert _variant_names(tmp_path, 8) == []
        seen.append(True)
        return original_emit(*args, **kwargs)

    monkeypatch.setattr(cli, 'emit', inspect_pending)
    monkeypatch.setattr(cli.shutil, 'copy2', inspect_copy)
    code, record = invoke(package, tmp_path, cli=cli)
    assert code == 0 and len(seen) == 2 and copied
    assert record['status'] == 'APPLIED_EMITTED_CHANGE'
    assert (tmp_path / 'application.json').is_file()
    assert not (tmp_path / 'variants/.library-pass-pending-changed').exists()
    assert _variant_names(tmp_path, 8) == ['changed']
