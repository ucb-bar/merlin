"""Statically re-admit a full Smol ELF with the pinned instruction assembler.

No device Core is constructed and no guest or accelerator program executes.
"""

import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tools/host_iss')]
import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked
from atlas_tooling import sha256, write_json
from full_bundle import FullBundle
from elf_image import Image
from smol_model_binding import configure_model, model_binding
import smol_full_contract
from kernel_library import code_sources


def fusion_source_pins(provenance):
    """Pin opt-in recipe code before FullBundle executes its fused validators."""
    fused = (provenance.get('fusion_set', 'none') != 'none'
             or any(isinstance(row, dict) and row.get('fusion') for row in provenance.get('kernels', ()))
             or bool(provenance.get('call_views')))
    if not fused:
        return {}
    paths = [ROOT / kernel_library.relative_path(name)
             for name in ('smol_fusion_assets.py', 'smol_fusion_program.py', 'smol_fusion_fp8.py')]
    return {str(checked(path)): sha256(path) for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'model-root', 'out'):
        parser.add_argument('--'+name, type=checked, required=True)
    args = parser.parse_args()
    if not __debug__:
        raise ValueError('static admission requires Python assertions')
    if args.out.exists() or args.out.is_relative_to(args.bundle):
        raise ValueError('fresh audit receipt outside the bundle required')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    receipt = dict(schema='atlas.smol_full_static_audit.v1', status='FAILED',
                   command=[sys.executable, *sys.argv], python=sys.version,
                   host_elf_executed=False, model_core_executed=False, hardware_execution=False)
    try:
        # This preliminary read chooses only the conservative source closure.
        # FullBundle independently authenticates and validates the metadata.
        pins = fusion_source_pins(json.loads(checked(args.bundle / 'provenance.json').read_bytes()))
        bundle = FullBundle(args.bundle, 'smol-full')
        files = [Path(__file__).absolute(), ROOT / 'tools/smol_full_plan.py']
        for name in ('atlas_path_guard', 'atlas_tooling', 'full_bundle', 'block_bundle', 'bundle', 'placement',
                     'elf_image', 'run_model_tile', 'smol_model_binding', 'smol_attention_assets', 'atlas_attention_v2_program', 'smol_full_contract', 'chipyard.arena'):
            files.append(checked(importlib.import_module(name).__file__))
        files.extend(ROOT / kernel_library.relative_path(name) for name in code_sources(traffic_reuse=True))
        pins.update({str(checked(path)): sha256(path) for path in files})
        _, _, assemble, _ = configure_model(args.model_root)
        before = model_binding(args.model_root)
        elf_raw = bundle.read('host/host.elf')
        data = smol_full_contract.validate_bundle(bundle, Image(elf_raw), assemble=assemble)
        bundle.verify_bookends()
        if model_binding(args.model_root) != before or any(sha256(path) != value for path, value in pins.items()):
            raise ValueError('static admission model/source bookends changed')
        receipt.update(status='PASS_STATIC_ONLY', elf_sha256=bundle.sums['host/host.elf'],
                       bundle_sha256sums_sha256=bundle.checksums_sha256,
                       contract_sha256=bundle.provenance['contract_sha256'],
                       kernel_count=bundle.kernel_count, projection_calls=data['projection_calls'],
                       source_sha256=pins, model_binding=before, source_bookends_equal=True,
                       baseline_programs_exact_captured_relocations=True,
                       reassembled_reuse_program_variants=len(bundle._smol_verified_programs))
    except Exception as error:
        receipt['error'] = dict(type=type(error).__name__, message=str(error))
    write_json(args.out, receipt)
    print(json.dumps({key: receipt[key] for key in ('status', 'error') if key in receipt}), flush=True)
    return 0 if receipt['status'] == 'PASS_STATIC_ONLY' else 1


if __name__ == '__main__':
    raise SystemExit(main())
