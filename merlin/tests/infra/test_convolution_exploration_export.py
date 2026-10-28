"""Prospective normal V22 selection exports executable registry dependencies."""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.compile.scheduling.api_catalog import scheduling_capabilities
from merlin.compile.scheduling.library_sources import library_modules
from merlin.targetgen.compiler_library import freeze_compiler_library
from merlin.targetgen.tool_profile import _SHARED_SUBSTRATES, COMMON_SUBSTRATE_V14, COMMON_SUBSTRATE_V15
from merlin.targetgen.tool_profile_shared import prepare_shared_source_release, reviewed_shared_manifest


def test_v22_normal_preparation_and_isolated_real_registry(tmp_path):
    additions = ('merlin.compile.logical_abi', 'merlin.compile.scheduling.convolution_exploration',
                 'merlin.compile.scheduling.dense_rhs_residency',
                 'merlin.transforms.pipeline_policy', 'merlin.transforms.pass_exploration',
                 'merlin.transforms.evaluated_exploration', 'merlin.transforms.shape_policy',
                 'merlin.compile.scheduling.dense_panel_epochs')
    assert library_modules(22) == (*library_modules(21), *additions)
    assert _SHARED_SUBSTRATES[COMMON_SUBSTRATE_V14] == 21
    assert _SHARED_SUBSTRATES[COMMON_SUBSTRATE_V15] == 22
    assert {c['name'] for c in scheduling_capabilities(22)['capabilities']} >= {
        'registered_source_convolution_exploration', 'selected_logical_kernel_argument_binding'}
    root = tmp_path/'sdk'
    manifest = prepare_shared_source_release(root, python_source_root=repo_root()/'src',
        license_directory=repo_root(), version=22, source_binding=True)
    assert reviewed_shared_manifest(root, version=22) == manifest
    with pytest.raises(ValueError):
        reviewed_shared_manifest(root, version=21)
    contract = freeze_compiler_library(root, review_id='explicit-spatial-registry-api',
        public_modules=tuple(manifest['public_modules']),
        sources=tuple((r['path'], r['module']) for r in manifest['members'])
        + tuple((path, None) for path in manifest['review_resources']))
    contract.verify(root)
    declaration = yaml.safe_load((repo_root()/'merlin/contract/mlir_oot_backend_contract.yaml').read_text())
    declaration = {'logical_kernel_abi': declaration['logical_kernel_abi']}
    from test_public_logical_abi import implicit_commit

    buffer = implicit_commit()
    code = '\n'.join([
        'import pathlib,sys', f'root=pathlib.Path({str(root)!r})',
        'sys.path.insert(0,str(root))',
        f'sys.path.append({str(Path(__file__).parents[1]/"ir")!r})',
        'import test_shape_policy as shape',
        'shape.test_registered_convolution_policy_emits_current_geometry_not_training_bytes()',
        'import test_shape_policy_proposals as proposals',
        'proposals.test_propose_validate_and_execute_unseen_current_ir_without_repeating_complete_panel()',
        'proposals.test_explicit_rational_feature_assessment_keeps_exact_boundary_not_integer_truncation()',
        'import test_evaluated_exploration as observations',
        'observations.test_real_registered_transform_dedup_and_exact_evaluated_product_selected()',
        'import test_convolution_exploration as check',
        'check.test_registry_realizes_only_source_bound_programs_and_deduplicates_equivalent_capacity()',
        'import test_convolution_rhs_residency as rhs',
        'rhs.test_registry_preparation_is_optional_and_capacity_fallback_deduplicates_actual_bytes()',
        'import test_convolution_compute_epochs as epochs',
        'epochs.test_epoch_registry_is_optional_and_realized_programs_are_bound_and_deduplicated()',
        'epochs.test_epoch_fence_is_inside_actual_counted_llvm_control()',
        'from merlin.compile.logical_abi import logical_abi_from_document,logical_interface',
        'from merlin.llvmlower.host_builder import FnBuilder',
        f'abi=logical_abi_from_document({declaration!r})',
        f'cb={buffer!r}',
        'builder,pointers=FnBuilder.for_logical_kernel(cb,abi=abi)',
        'assert list(pointers)==["weights","activations","result"]',
        'assert pointers["activations"] is builder.entry.args[1]',
        'assert logical_interface(cb,abi)[-1].shape==(3,5)',
        'cb["tensors"]["activations"]["shape"]=[12,6]',
        'cb["commands"][-1]["attributes"].update(epilogue=["maxpool"],pool_in_dims=[3,4],'
        'pool_size=[2,2],pool_stride=[1,1],pool_padding=[0,0,0,0])',
        'assert logical_interface(cb,abi)[-1].shape==(6,5)',
        'assert not any(n in sys.modules for n in ("merlin.runtime.commandbuffer",'
        '"merlin.runtime.tensor","merlin.runtime.reference","merlin.targetgen.contract.harness_render"))',
        'for name,owner in tuple(sys.modules.items()):',
        ' if name=="merlin" or name.startswith("merlin."):',
        '  path=getattr(owner,"__file__",None)',
        '  if path: assert pathlib.Path(path).is_relative_to(root),(name,path)',
        'print("spatial-registry-export-PASS")',
    ])
    result = subprocess.run([sys.executable, '-I', '-B', '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'spatial-registry-export-PASS'
    leaf = root/'merlin/compile/scheduling/convolution_exploration.py'
    leaf.write_text(leaf.read_text()+'\n# altered selected source\n')
    with pytest.raises(ValueError):
        contract.verify(root)
