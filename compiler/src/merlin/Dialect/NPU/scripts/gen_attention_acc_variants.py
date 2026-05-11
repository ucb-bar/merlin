"""Regenerate manifest entries for attention_acc_first/mid/last.

Calls the body generators in
``benchmarks/SaturnNPU/kernel_library/attention_acc_kernels.py`` (which
derive from the verified ``smolvla_fused_attention.py`` reference) and
writes them to ``manifest.json``.

Run after any edit to ``attention_acc_kernels.py``, then re-run
``annotate_kernel_patch_points.py`` to repopulate dram_in/out offsets.
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
MANIFEST = REPO / "benchmarks" / "SaturnNPU" / "kernel_library" / "manifest.json"


def _instruction_to_dict(instruction) -> dict:
    args = instruction.args
    return {
        "mnemonic": instruction.mnemonic,
        "args": asdict(args) if is_dataclass(args) else dict(args),
    }


def main() -> int:
    sys.path.insert(0, str(REPO / "third_party" / "npu_model"))
    sys.path.insert(0, str(REPO))

    # Heavy imports (torch via npu_model.isa) live inside main so "--help"
    # etc. stay cheap.
    import npu_model  # noqa: F401
    from npu_model.configs.programs import _attention_acc_kernels as ak

    manifest = json.loads(MANIFEST.read_text())
    if "attention" not in manifest["kernels"]:
        print("ERROR: base 'attention' kernel missing from manifest", file=sys.stderr)
        return 1

    # Reuse tile_shape / layout annotations from the single-tile attention
    # kernel — the new variants operate on the same per-tile Q/K/V/scale
    # shapes.
    base = manifest["kernels"]["attention"]
    base_meta = {
        k: v
        for k, v in base.items()
        if k
        not in {
            "instructions",
            "patch_points",
            "num_instructions",
            "cycles",
            "symbol_prefix",
            "source_file",
            "status",
            "status_note",
        }
    }

    def _register(name: str, body_fn) -> None:
        instructions = [_instruction_to_dict(i) for i in body_fn()]
        entry = copy.deepcopy(base_meta)
        entry["symbol_prefix"] = f"npu_uk_attention_acc_{name}"
        entry["source_file"] = "kernel_library/attention_acc_kernels.py"
        entry["num_instructions"] = len(instructions)
        # Rough cycle estimate: ~500 cycles per variant (one KV iter + state
        # I/O). The annotator doesn't rely on this; it's informational.
        entry["cycles"] = 500
        entry["patch_points"] = []  # repopulated by annotate_kernel_patch_points.py
        entry["instructions"] = instructions
        manifest["kernels"][f"attention_acc_{name}"] = entry

    _register("first", ak.first_body)
    _register("mid", ak.mid_body)
    _register("last", ak.last_body)

    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    for name in ("first", "mid", "last"):
        entry = manifest["kernels"][f"attention_acc_{name}"]
        print(f"  attention_acc_{name}: {entry['num_instructions']} insts")
    print(f"wrote real bodies to {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
