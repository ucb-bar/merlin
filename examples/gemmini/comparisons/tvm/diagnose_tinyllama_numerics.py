"""Host-arithmetic ablations on the local legacy TinyLlama W8A8 recipe.

This is a diagnostic, not a selected EL4 oracle or an emulator of a remote ELF.
Weights are streamed one layer at a time. Integer projections use the existing
reference in every arm; only the explicitly named floating operations change.
TVM kernels are imported from torch.export and executed by the host LLVM VM.
"""

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from tinyllama_integer import IntegerProjection, diagnose_activation_pair, recipe

MODE_SCOPE = {
    "native": "PyTorch eager legacy recipe",
    "rms": "TVM RMSNorm only; all other operations retain the native recipe",
    "silu": "TVM SiLU only; all other operations retain the native recipe",
    "exp_trig": "TVM sin/cos/exp plus explicit exp/sum/divide softmax and divide-form SiLU",
    "portable": "Shared C sin/cos/exp plus the same explicit softmax and divide-form SiLU",
    "all": "TVM RMSNorm, SiLU, softmax, QK/PV and sin/cos; projections and rotary algebra remain PyTorch",
}


def compare(reference, actual):
    """Report elementwise agreement without accepting nonfinite pairs."""
    a, b = np.ascontiguousarray(reference), np.ascontiguousarray(actual)
    if a.shape != b.shape or a.dtype != b.dtype or not a.size:
        raise ValueError("Comparison requires equal nonempty shapes and dtypes")
    finite = np.isfinite(a) & np.isfinite(b)
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    close = finite & (delta <= 2e-6 + 1e-5 * np.abs(a.astype(np.float64)))
    if a.dtype.kind not in "fc":
        close = a == b
    changed = np.any(
        a.view(np.uint8).reshape(-1, a.dtype.itemsize) != b.view(np.uint8).reshape(-1, b.dtype.itemsize), axis=-1
    )
    worst = int(np.argmax(np.where(finite, delta, np.inf)))
    return {
        "shape": list(a.shape),
        "count": a.size,
        "nonfinite": int((~finite).sum()),
        "bit_mismatches": int(changed.sum()),
        "outside_tolerance": int((~close).sum()),
        "max_abs": float(delta[finite].max()) if finite.any() else None,
        "worst_index": list(np.unravel_index(worst, a.shape)),
        "reference_at_worst": float(a.flat[worst]),
        "actual_at_worst": float(b.flat[worst]),
    }


class RMS(torch.nn.Module):
    def forward(self, x, weight):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5) * weight


class SiLU(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.silu(x)


class Softmax(torch.nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=-1)


class Matmul(torch.nn.Module):
    def forward(self, a, b):
        return a @ b


class Unary(torch.nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, x):
        return getattr(torch, self.op)(x)


class HostKernels:
    def __init__(self, output):
        import tvm

        self.tvm, self.output, self.cache = tvm, output, {}

    def __call__(self, name, *values):
        from tvm import relax
        from tvm.relax.frontend.torch import from_exported_program

        key = (name, *(tuple(x.shape) for x in values))
        if key not in self.cache:
            module = {"rms": RMS, "silu": SiLU, "softmax": Softmax, "matmul": Matmul}.get(name)
            module = module() if module else Unary(name)
            exported = torch.export.export(module, values)
            mod = from_exported_program(exported, unwrap_unit_return_tuple=True)
            stem = f"kernel-{len(self.cache):02d}-{name}"
            (self.output / f"{stem}.relax.py").write_text(mod.script())
            exe = relax.build(mod, target="llvm")
            self.cache[key] = relax.VirtualMachine(exe, self.tvm.cpu())
        result = self.cache[key]["main"](*(self.tvm.nd.array(x.contiguous().numpy()) for x in values))
        return torch.from_numpy(result.numpy())


def rotate(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sin


def identity(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": sha}


def serializable(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


@torch.no_grad()
def run(args, output):
    import tinyllama_integer
    from tinyllama_session import WEIGHT_SHA256

    checkpoint = identity(args.checkpoint)
    if checkpoint["sha256"] != WEIGHT_SHA256:
        raise ValueError("Diagnostic geometry requires the pinned full TinyLlama checkpoint")
    sources = {"script": identity(__file__), "integer_reference": identity(tinyllama_integer.__file__)}
    torch.set_num_threads(1)
    kernels = HostKernels(output)
    from merlin.runtime.host_math_reference import reference_math

    portable = reference_math("portable_exp_trig", output / "portable")
    tokens = json.loads(args.tokens_json.read_text())["v00"]["tokens"][: args.tokens]
    if len(set(args.modes)) != len(args.modes):
        raise ValueError("Ablation modes must be unique")
    modes = ["native", *args.modes]
    states, records, first_codes, probes = {}, [], {}, []
    frequencies = 1.0 / (10000.0 ** (torch.arange(0, 64, 2).float() / 64))
    angles = torch.arange(args.tokens).float()[:, None] * frequencies[None, :]
    angles = torch.cat((angles, angles), -1)[None, None, :, :]
    native_rope = (angles.cos(), angles.sin())
    tvm_rope = (kernels("cos", angles), kernels("sin", angles))
    portable_rope = tuple(torch.from_numpy(portable.evaluate("math." + op, angles.numpy())) for op in ("cos", "sin"))
    for op, ref, actual, fixed in zip(("cos", "sin"), native_rope, tvm_rope, portable_rope):
        probes.append(
            {
                "operation": op,
                "tvm_vs_torch": compare(ref.numpy(), actual.numpy()),
                "portable_vs_torch": compare(ref.numpy(), fixed.numpy()),
            }
        )
    mask = torch.full((args.tokens, args.tokens), torch.finfo(torch.float32).min).triu(1)
    with safe_open(args.checkpoint, framework="pt", device="cpu") as tensors:
        embedding = tensors.get_tensor("model.embed_tokens.weight")[tokens].float()[None]
        states = {mode: embedding.clone() for mode in modes}
        for layer in range(args.layers):
            prefix = f"model.layers.{layer}."
            weights = {
                name: tensors.get_tensor(prefix + name + ".weight").float()
                for name in ("input_layernorm", "post_attention_layernorm")
            }
            projections = {
                name: IntegerProjection(
                    tensors.get_tensor(prefix + name + ".weight").float(), activation_scale=args.activation_scale
                )
                for name in (
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                    "mlp.gate_proj",
                    "mlp.up_proj",
                    "mlp.down_proj",
                )
            }
            native_trace = {}
            for mode in modes:

                def observe(name, value, projection=None):
                    value = value.contiguous()
                    if mode == "native":
                        native_trace[name] = value.clone()
                    else:
                        row = {
                            "layer": layer,
                            "mode": mode,
                            "operation": name,
                            **compare(native_trace[name].numpy(), value.numpy()),
                        }
                        if projection is not None:
                            row["quantization"] = diagnose_activation_pair(
                                native_trace[name].numpy(), value.numpy(), args.activation_scale
                            )
                            if row["quantization"]["code_mismatches"] and mode not in first_codes:
                                first_codes[mode] = row
                        records.append(row)
                    return value

                def project(name, x):
                    return projections[name](observe(name + ".input", x, projections[name]))

                def norm(x, weight):
                    if mode in ("rms", "all"):
                        return kernels("rms", x, weight)
                    return RMS()(x, weight)

                x = states[mode]
                h = observe("input_norm", norm(x, weights["input_layernorm"]))
                q = project("self_attn.q_proj", h).reshape(1, args.tokens, 32, 64).transpose(1, 2)
                k = project("self_attn.k_proj", h).reshape(1, args.tokens, 4, 64).transpose(1, 2)
                v = project("self_attn.v_proj", h).reshape(1, args.tokens, 4, 64).transpose(1, 2)
                rope = tvm_rope if mode in ("exp_trig", "all") else portable_rope if mode == "portable" else native_rope
                q, k = rotate(q, *rope), rotate(k, *rope)
                observe("key", k)
                observe("value", v)
                k, v = k.repeat_interleave(8, 1), v.repeat_interleave(8, 1)
                scores = (
                    kernels("matmul", q, k.transpose(-1, -2)) if mode == "all" else q @ k.transpose(-1, -2)
                ) * 0.125 + mask
                observe("attention_scores", scores)
                if mode in ("exp_trig", "portable"):
                    # Separate explicit-exp formula from native fused softmax.
                    shifted = scores - scores.amax(-1, keepdim=True)
                    exp = (
                        kernels("exp", shifted)
                        if mode == "exp_trig"
                        else torch.from_numpy(portable.evaluate("math.exp", shifted.numpy()))
                    )
                    prob = exp / exp.sum(-1, keepdim=True)
                else:
                    prob = kernels("softmax", scores) if mode == "all" else torch.softmax(scores, -1)
                observe("probabilities", prob)
                attn = kernels("matmul", prob, v) if mode == "all" else prob @ v
                attn = attn.transpose(1, 2).reshape(1, args.tokens, 2048)
                x = observe("attention_residual", x + project("self_attn.o_proj", attn))
                h = observe("post_attention_norm", norm(x, weights["post_attention_layernorm"]))
                gate, up = project("mlp.gate_proj", h), project("mlp.up_proj", h)
                if mode in ("silu", "all"):
                    activated = kernels("silu", gate)
                elif mode in ("exp_trig", "portable"):
                    exp = (
                        kernels("exp", -gate)
                        if mode == "exp_trig"
                        else torch.from_numpy(portable.evaluate("math.exp", -gate.numpy()))
                    )
                    activated = gate / (1 + exp)
                else:
                    activated = torch.nn.functional.silu(gate)
                if layer == 0 and mode == "native":
                    # Same inputs and same exp isolate the SiLU operation ordering.
                    exp = torch.exp(-gate)
                    probes.append(
                        {
                            "operation": "silu_same_exp_divide_vs_reciprocal_multiply",
                            **compare((gate / (1 + exp)).numpy(), (gate * (1 / (1 + exp))).numpy()),
                        }
                    )
                    probes.append(
                        {
                            "operation": "silu_tvm_same_input",
                            **compare(activated.numpy(), kernels("silu", gate).numpy()),
                            "exp_tvm_vs_torch": compare(exp.numpy(), kernels("exp", -gate).numpy()),
                        }
                    )
                observe("silu", activated)
                states[mode] = observe("layer_output", x + project("mlp.down_proj", activated * up))
            print(
                json.dumps(
                    {
                        "layer": layer,
                        "first_code_modes": list(first_codes),
                        "layer_max_abs": {
                            mode: compare(states["native"].numpy(), states[mode].numpy())["max_abs"]
                            for mode in args.modes
                        },
                    }
                ),
                flush=True,
            )
            del projections, weights, native_trace
        logits = {}
        if args.layers == 22:
            weight = tensors.get_tensor("model.norm.weight").float()
            head = IntegerProjection(tensors.get_tensor("lm_head.weight").float(), args.activation_scale)
            for mode, state in states.items():
                normalized = kernels("rms", state, weight) if mode in ("rms", "all") else RMS()(state, weight)
                logits[mode] = head(normalized[:, -1:, :]).numpy()
    np.savez(
        output / "outputs.npz",
        **{f"hidden_{k}": v.numpy() for k, v in states.items()},
        **{f"logits_{k}": v for k, v in logits.items()},
    )
    from tvm._ffi.base import _LIB

    result = {
        "scope": "host_arithmetic_ablation_of_legacy_dynamic_w8a8_recipe_not_remote_ELF_or_selected_EL4",
        "full_model_qualified": False,
        "tokens": tokens,
        "layers": args.layers,
        "recipe": recipe(args.activation_scale),
        "rtol": 1e-5,
        "atol": 2e-6,
        "modes": modes,
        "mode_scope": {mode: MODE_SCOPE[mode] for mode in modes},
        "checkpoint": checkpoint,
        "token_source": identity(args.tokens_json),
        "sources": sources,
        "torch": torch.__version__,
        "cpu": torch.backends.cpu.get_cpu_capability(),
        "python": platform.python_version(),
        "tvm_source": kernels.tvm.__file__,
        "tvm_library": identity(_LIB._name),
        "tvm_commit": subprocess.check_output(
            ["git", "-C", str(Path(kernels.tvm.__file__).parent), "rev-parse", "HEAD"], text=True
        ).strip(),
        "limitations": [
            "prefill only; no decode/cache recurrence",
            "host LLVM VM, not remote RISC-V ELF",
            "legacy dynamic quantization, not selected EL4 static calibrated recipe",
            "integer projections/quantizers are shared PyTorch references in all arms",
            "exp_trig/portable arms also replace fused softmax by an explicit formula",
        ],
        "portable_reference": portable.identity(),
        "probes": probes,
        "first_changed_codes": first_codes,
        "logits": {
            mode: {**compare(logits["native"], value), "top5": np.argsort(value.ravel())[-5:][::-1].tolist()}
            for mode, value in logits.items()
        },
        "checkpoints": records,
    }
    (output / "report.json").write_text(json.dumps(result, indent=2, default=serializable) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokens-json", type=Path, required=True)
    parser.add_argument("--tokens", type=int, choices=(16, 32), default=16)
    parser.add_argument("--layers", type=int, choices=range(1, 23), default=22)
    parser.add_argument("--activation-scale", choices=("power_of_two", "maxabs"), default="power_of_two")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("rms", "silu", "exp_trig", "portable", "all"),
        default=["rms", "silu", "exp_trig", "portable", "all"],
    )
    args = parser.parse_args()
    from merlin.common.artifacts import new_product

    product = new_product("audits", version=1, target="gemmini-tvm-numerics", notes=__doc__)
    print(product.path, flush=True)
    result = run(args, product.path)
    product.add_artifact("report.json")
    product.add_artifact("outputs.npz")
    product.write_manifest()
    print(json.dumps({"logits": result["logits"]}, default=serializable), flush=True)


if __name__ == "__main__":
    main()
