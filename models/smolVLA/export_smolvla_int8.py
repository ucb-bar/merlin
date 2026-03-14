#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
UNDERSTANDING_PI0_ROOT = REPO_ROOT / "third_party" / "Understanding-PI0"
if str(UNDERSTANDING_PI0_ROOT) not in sys.path:
    sys.path.insert(0, str(UNDERSTANDING_PI0_ROOT))

from understanding_pi0.common.env import (  # noqa: E402
    print_runtime_info,
    seed_all,
    warn_if_mx_execution_unavailable,
)
from understanding_pi0.common.iree_ocp_patch import apply_all_iree_ocp_patches  # noqa: E402
from understanding_pi0.common.mx_exportable import (  # noqa: E402
    clone_and_rewrite_quantized_linears_for_export,
)
from understanding_pi0.common.torchao_utils import safe_quantize_linears_  # noqa: E402
from understanding_pi0.smolvla_mx.loader import (  # noqa: E402
    build_dummy_processed_inputs,
    load_smolvla_policy,
)
from understanding_pi0.smolvla_mx.wrappers import (  # noqa: E402
    SmolVLAOneStepNoCacheWrapper,
    flatten_processed_inputs,
)


def save_mlir_fallback(exported, out_path: Path) -> None:
    if hasattr(exported, "save_mlir"):
        exported.save_mlir(str(out_path))
        return
    if hasattr(exported, "mlir_module"):
        out_path.write_text(str(exported.mlir_module), encoding="utf-8")
        return
    if hasattr(exported, "module"):
        out_path.write_text(str(exported.module), encoding="utf-8")
        return
    raise RuntimeError("Could not find a way to save MLIR from the exported object.")


def build_int8_plan(model, quantize_vision: bool = True) -> OrderedDict[str, str | None]:
    plan: OrderedDict[str, str | None] = OrderedDict()
    for fqn, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if "lm_head" in fqn:
            plan[fqn] = None
            continue
        if not quantize_vision and ("vision_model" in fqn or ".connector." in fqn):
            plan[fqn] = None
            continue
        plan[fqn] = "int8"
    return plan


def main() -> int:
    ap = argparse.ArgumentParser(description="Export SmolVLA one-step MLIR with forced int8 quantization.")
    ap.add_argument("--model-id", default="lerobot/smolvla_base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--image-h", type=int, default=256)
    ap.add_argument("--image-w", type=int, default=256)
    ap.add_argument("--prompt-len", type=int, default=8)
    ap.add_argument(
        "--mx-kernel-preference",
        default="AUTO",
        help="Accepted for CLI compatibility with mixed export path; ignored in forced-int8 mode.",
    )
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--skip-patches", action="store_true")
    ap.add_argument("--no-exportable-mx", action="store_true")
    ap.add_argument("--out", default="models/smolVLA/smolVLA.q.int8.mlir")
    args = ap.parse_args()

    seed_all(args.seed)
    print_runtime_info(args.device)
    warn_if_mx_execution_unavailable(args.device)

    if not args.skip_patches:
        apply_all_iree_ocp_patches(verbose=True)

    import iree.turbine.aot as aot

    policy = load_smolvla_policy(args.model_id, device=args.device).to(torch.bfloat16).eval()
    plan = build_int8_plan(policy, quantize_vision=not args.no_vision)
    _ = safe_quantize_linears_(
        policy,
        plan=plan,
        quant_device=args.device,
        mx_kernel_preference=None,
        verbose=False,
    )

    if not args.no_exportable_mx:
        policy, records = clone_and_rewrite_quantized_linears_for_export(
            policy,
            compute_dtype=torch.bfloat16,
            verbose=False,
        )
        n_replaced = sum(int(r.replaced) for r in records)
        print(f"[exportable_linear] replaced {n_replaced} quantized linears for export")

    sample = build_dummy_processed_inputs(
        policy,
        batch_size=args.batch_size,
        image_hw=(args.image_h, args.image_w),
        prompt_len=args.prompt_len,
        device=args.device,
    )
    wrapper = SmolVLAOneStepNoCacheWrapper(policy, num_cams=len(sample["images"])).eval()
    example_args = flatten_processed_inputs(sample)

    print("[export:int8] running iree.turbine.aot.export(...)")
    exported = aot.export(
        wrapper,
        args=example_args,
        strict_export=False,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_mlir_fallback(exported, out_path)
    print(f"[saved] int8 mlir -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
