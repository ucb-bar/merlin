"""The ordinary dense backend can try checked advanced families without copying recipes."""
from __future__ import annotations

import subprocess
import sys

import pytest
from author_skill_fixtures import load_script, skill_dir

from merlin.common.paths import repo_root
from merlin.compile.scheduling import dense_ready_readout, schedule_ir
from merlin.targetgen.tool_profile_shared import prepare_shared_source_release, reviewed_shared_manifest


def _llvm(result):
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import ModuleOp, i64

    from merlin.llvmlower.host_builder import FnBuilder

    builder = FnBuilder([])
    declarations = {}

    def lower(instruction, arguments, metadata=None):
        from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter

        bridge = FnBuilderIndexAdapter(builder)
        values = tuple(bridge.value(value) for value in arguments.values())
        assert instruction not in declarations or declarations[instruction] == len(values)
        declarations[instruction] = len(values)
        builder.add(llvm.CallOp(instruction, *values))

    result.emit_llvm(builder, lower)
    module = ModuleOp([*(llvm.FuncOp(n, llvm.LLVMFunctionType([i64] * count, llvm.LLVMVoidType()))
                        for n, count in declarations.items()),
                       llvm.FuncOp('actual', llvm.LLVMFunctionType([], llvm.LLVMVoidType()), body=builder.finish())])
    module.verify()
    return module, declarations


@pytest.mark.parametrize('shape', [(12, 12, 12), (28, 12, 4), (4, 12, 28), (1, 12, 28), (7, 13, 11)])
def test_actual_square_tall_wide_thin_and_ragged_routes_preserve_products_and_completion(shape, monkeypatch):
    example = load_script('shape-schedule-selection', 'capacity_pipeline_example')
    captured = []
    original = dense_ready_readout.schedule_dense_pipeline

    def consume(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(dense_ready_readout, 'schedule_dense_pipeline', consume)
    # The tall N=one-tile shape has no disjoint N group. It keeps compact/retained
    # instead of inventing a readout-overlap grant or silently changing block.
    routes = (('compact', 'retained') if shape[-1] == 4 else
              ('compact', 'segmented', 'ready') if shape[0] <= 4 else
              ('compact', 'retained', 'segmented', 'ready'))
    report = example.example(shape=shape, pipelines=routes)
    assert set(report['pipelines']) == set(routes)
    assert len(captured) == len(routes)
    for route, result in zip(routes, captured, strict=True):
        result.revalidate()
        from merlin.compile.scheduling.scf_commands import materialize_scf_commands

        materialize_scf_commands(result.kernel).verify()
        module, declarations = _llvm(result)
        assert {'lhs_load', 'weight_load', 'preload', 'compute', 'store'} <= declarations.keys()
        if route != 'compact':
            assert 'merlin_dense_panel_drain' in declarations
        # Compact static representation; no expansion by dynamic tensor shape.
        assert len(tuple(module.walk())) < 1500
    if shape[-1] == 4:
        with pytest.raises(SystemExit, match='no capacity-legal block'):
            example.example(shape=shape, pipelines=('segmented',))


def test_missing_wrong_or_unused_execution_laws_refuse_without_fallback():
    example = load_script('shape-schedule-selection', 'capacity_pipeline_example')
    caps = example.capacities(None)
    p, s, r, _ = example.shared_upstream(example.derived_shape(caps), caps)
    a = example.adapter()
    common = dict(name='test', dtypes=example.DTYPES, external_buffers_disjoint=True)
    with pytest.raises(ValueError, match='cannot consume'):
        dense_ready_readout.schedule_dense_pipeline(p, r, s, a, pipeline_admission=object(), **common)
    with pytest.raises(ValueError, match='readout mode'):
        dense_ready_readout.schedule_dense_pipeline(p, r, s, a, readout_admission=object(), **common)
    from dataclasses import replace

    from merlin.transforms.dense_planning import dense_blocks

    r = replace(r, max_k_blocks=1)
    b = dense_blocks(p, r)[0]
    plan = schedule_ir.make_dense_schedule_plan(p, r, s, a, source_identity=example.SOURCE,
        dtypes=example.DTYPES, static_budget=128, block_tiles=(b.m, b.k, b.n),
        pipeline_kind='two_window', pipeline_prefetch=False)
    with pytest.raises(ValueError, match='independently selected pipeline law'):
        dense_ready_readout.schedule_dense_pipeline(p, r, s, a, schedule_plan=plan,
                                           source_identity=example.SOURCE, **common)
    with pytest.raises(ValueError, match='binding differs'):
        dense_ready_readout.schedule_dense_pipeline(p, r, s, a, schedule_plan=plan,
                                           source_identity='d'*64, **common)


def test_normal_sdk_isolated_recipe_import_and_all_four_actual_routes(tmp_path):
    sdk = tmp_path/'sdk'
    manifest = prepare_shared_source_release(sdk, python_source_root=repo_root()/'src',
        license_directory=repo_root(), version=22, source_binding=True)
    assert reviewed_shared_manifest(sdk, version=22) == manifest
    script = skill_dir('shape-schedule-selection')/'scripts/capacity_pipeline_example.py'
    code = '\n'.join([
        'import importlib.util,pathlib,sys',
        f'root=pathlib.Path({str(sdk)!r})', 'sys.path.insert(0,str(root))',
        f'spec=importlib.util.spec_from_file_location("recipe",{str(script)!r})',
        'recipe=importlib.util.module_from_spec(spec);spec.loader.exec_module(recipe)',
        'from merlin.transforms.dense_planning import DenseProblem,DenseResources',
        'from merlin.compile.scheduling.compact_dense import DenseLoopStorage,DenseMemoryView',
        'p=DenseProblem(7,12,11,4,4,4,1,1,4)',
        's=DenseLoopStorage(DenseMemoryView("lhs","operand",4,0,64),'
        'DenseMemoryView("rhs","operand",4,64,64),DenseMemoryView("acc","accumulator",16,0,32))',
        'r=DenseResources(512,512,max_m_blocks=2,max_n_blocks=3,max_k_blocks=3,'
        'partial_output_tiles=True,partial_output_blocks=True)',
        'out={name:f(p,s,r,recipe.adapter()) for name,f in recipe.PIPELINES.items()}',
        'assert set(out)=={"compact","retained","segmented","ready"}',
        'from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline',
        'assert pathlib.Path(sys.modules[schedule_dense_pipeline.__module__].__file__).is_relative_to(root)',
        'for name,owner in tuple(sys.modules.items()):',
        ' if name=="merlin" or name.startswith("merlin."):',
        '  path=getattr(owner,"__file__",None)',
        '  if path: assert pathlib.Path(path).is_relative_to(root),(name,path)',
        'print("actual-advanced-routes-PASS")',
    ])
    result = subprocess.run([sys.executable, '-I', '-B', '-c', code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'actual-advanced-routes-PASS'
