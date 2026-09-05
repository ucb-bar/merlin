#!/usr/bin/env python3
"""Write an **INDEPENDENT** W8A8 reference for an int8 capture bundle.

Why this exists, and why it is a separate tool from ``make_w8a8_golden.py``.

``make_w8a8_golden.py`` computes ``golden_w8a8.npy`` with *merlin's own* int8 datapath
(``dispatch_runtime.run_model(int8_compute=True)``). That reference decides "did the device
reproduce what the host compiler computes" and nothing else: a host run scores ``cos 1.0 /
rel 0.0`` against it no matter what the arithmetic does, because the two sides are the same
program. Measured 2026-09-04: re-running ``spectformer_int8_full`` with the PRE-FIX
``passes_quant_int`` reproduced its shipped ``golden_w8a8.npy`` BIT-FOR-BIT (maxabs 0.0), while
the post-fix code differs by 0.0755 — i.e. that golden is a photograph of our own runtime frozen
at the 2026-08-18 code, and a W8A8 tier pass against it is not evidence about the arithmetic.

This tool produces the other kind: a reference computed OUTSIDE the compiler, by torchao's
``int8_dyn_act_int8_weight`` on the same seeded model instance the bundle was captured from,
executed in torch eager. Against it, ``w8a8_cos``/``w8a8_rel`` mean what the tier claims they
mean.

**FAIL CLOSED ON INCONSISTENCY — this is the whole value of the artifact.** The activation-
quantized instance must reproduce the BUNDLE's own quantized weights *bit-for-bit*, or nothing
is written. A golden belonging to different weights than the bundle ships manufactures failures
(or, worse, passes) that have nothing to do with the datapath under test; the refusal is not a
convenience check, it is the reason the number can be cited at all.

Two further consistency properties, both fail-closed:
  * the reference is computed on the BUNDLE's own recorded inputs (``inputs.npz``), not on
    re-seeded ones, so it cannot drift from the bundle if a loader's RNG consumption changes;
  * the output shape must match the bundle's fp32 ``golden.npy``.

It never overwrites ``golden_w8a8.npy``. It writes ``golden_w8a8.independent.npy`` beside it
(plus a ``.provenance.json`` recording that this one IS independent), because board runs in
flight are graded against the existing file and changing a golden underneath a running grade is
itself a failure mode. Cutting over is a separate, deliberate act.

Runs the heavy half inside the model's own capture venv (torch + torchao + m2m), resolved from
``$MERLIN_M2M_DIR/workloads/<model>/capture.toml`` exactly as the capture driver does.

Usage:
    make_w8a8_independent_golden.py <bundle> [<bundle> ...]
    make_w8a8_independent_golden.py --list          # what could be generated, write nothing
    make_w8a8_independent_golden.py --out-name golden_w8a8.independent.npy <bundle>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np

#: torchao scheme that quantizes ACTIVATIONS as well as weights — the arithmetic an integer
#: datapath actually performs. Mirrors ``capture.py``'s ``int8_w8a8`` format; the weight-only
#: ``int8`` scheme the bundles were captured with is a different program.
ACT_QUANT_SCHEME = "int8_dyn_act_int8_weight"

DEFAULT_OUT_NAME = "golden_w8a8.independent.npy"

#: safetensors dtype tag -> numpy dtype, for the byte-level reader below. Only dtypes we can
#: compare as raw bytes are listed; anything else is skipped rather than guessed at.
_ST_DTYPE = {"I8": np.int8, "U8": np.uint8, "I16": np.int16, "I32": np.int32, "I64": np.int64,
             "F16": np.float16, "F32": np.float32, "F64": np.float64,
             "BF16": np.uint16, "F8_E4M3": np.uint8, "F8_E5M2": np.uint8, "BOOL": np.bool_}


# --------------------------------------------------------------------------- shared helpers
class SafeTensors(Mapping):
    """Read-only, **memory-mapped** view of a safetensors file (no safetensors package).

    Mapped rather than read: the consistency gate compares two whole captures, and a 5 GB
    bundle plus its 5 GB freshly-quantized twin does not fit in RAM beside the model that
    produced one of them. Mapping keeps the comparison a page-cache walk. The inner worker runs
    in the model venv and the driver in merlin's, so parsing the container here also keeps both
    sides on one implementation and adds a dependency to neither.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        with self.path.open("rb") as stream:
            header_len = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_len))
        self._base = 8 + header_len
        self._meta = {name: meta for name, meta in header.items()
                      if name != "__metadata__" and meta["dtype"] in _ST_DTYPE}
        self._map = np.memmap(self.path, dtype=np.uint8, mode="r")

    def __iter__(self):
        return iter(self._meta)

    def __len__(self) -> int:
        return len(self._meta)

    def dtype(self, name: str) -> np.dtype:
        return np.dtype(_ST_DTYPE[self._meta[name]["dtype"]])

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._meta[name]["shape"])

    def __getitem__(self, name: str) -> np.ndarray:
        meta = self._meta[name]
        start, end = meta["data_offsets"]
        raw = self._map[self._base + start:self._base + end]
        return raw.view(_ST_DTYPE[meta["dtype"]]).reshape(meta["shape"])


def read_safetensors(path: Path) -> SafeTensors:
    return SafeTensors(path)


def quantized_weight_diff(bundle_weights, fresh_weights) -> dict:
    """Compare the INTEGER weight tensors of two captures of the same model.

    Returns a report; ``report["ok"]`` is true only when at least one integer tensor is shared
    and every shared integer tensor is bit-for-bit equal. Both halves matter: an empty
    intersection is not agreement, it is an unmeasured comparison, and it must not pass.
    """
    shared = sorted(set(bundle_weights) & set(fresh_weights))
    integer = [k for k in shared
               if bundle_weights[k].dtype in (np.dtype(np.int8), np.dtype(np.uint8))
               and fresh_weights[k].dtype in (np.dtype(np.int8), np.dtype(np.uint8))]
    mismatched = [k for k in integer
                  if bundle_weights[k].shape != fresh_weights[k].shape
                  or not np.array_equal(bundle_weights[k], fresh_weights[k])]
    missing = sorted(k for k in bundle_weights if k not in fresh_weights)
    return {
        "n_bundle": len(bundle_weights),
        "n_fresh": len(fresh_weights),
        "n_shared": len(shared),
        "n_quantized": len(integer),
        "n_mismatched": len(mismatched),
        "mismatched_examples": mismatched[:5],
        "missing_from_fresh": missing[:5],
        "n_missing_from_fresh": len(missing),
        "ok": bool(integer) and not mismatched,
    }


def flatten_quantized_parameters(model) -> dict[str, np.ndarray]:
    """The ``qinner::`` leaf tensors of a model's quantized parameters, named exactly as
    ``capture_consistent.py`` names them (the ``__tensor_flatten__`` access chain).

    Some captures never put the integer weight in ``weights.safetensors`` at all: m2m leaves an
    uninitialized ``prov.quant_inner``-tagged empty there and the capture writes the real
    ``int_data``/``scale`` into ``extra.npz`` under ``qinner::<attr-path>`` (measured: the
    resnet50 W8A8 bundle ships ZERO int8 tensors in its safetensors and both of its integer
    tensors under ``qinner::``). Comparing only the safetensors on such a bundle finds an empty
    intersection — which is an unmeasured comparison, not agreement — so the gate needs to be
    able to read the other container too.
    """
    flat: dict[str, np.ndarray] = {}

    def walk(obj, prefix: str) -> None:
        flatten = getattr(obj, "__tensor_flatten__", None)
        if callable(flatten):
            try:
                names, _ = flatten()
            except Exception:                                           # noqa: BLE001
                names = []
            for name in names:
                child = getattr(obj, name, None)
                if child is not None:
                    walk(child, f"{prefix}.{name}")
        elif hasattr(obj, "detach"):
            tensor = obj.detach().cpu()
            text = str(tensor.dtype)
            flat[prefix] = (tensor.float().numpy()
                            if ("float8" in text or "bfloat16" in text) else tensor.numpy())

    for name, parameter in model.named_parameters():
        if type(parameter).__name__ not in ("Parameter", "Tensor") or hasattr(
                parameter, "__tensor_flatten__"):
            walk(parameter, name)
    return flat


def bundle_quantized_parameters(bundle: Path) -> dict[str, np.ndarray]:
    """The bundle's own ``qinner::`` entries, keyed the same way (prefix stripped)."""
    path = bundle / "extra.npz"
    if not path.is_file():
        return {}
    with np.load(path) as data:
        return {k[len("qinner::"):]: np.asarray(data[k])
                for k in data.files if k.startswith("qinner::")}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denom) if denom else float("nan")


def bundle_model_name(bundle: Path) -> str:
    """Capture-workload name for a recapture bundle directory.

    Bundle dirs are ``<model>_<dtype>_<variant>[...]``; the workload is the prefix before the
    dtype token. Derived by splitting on the dtype token that is actually present rather than
    by a fixed field count, because variants carry extra suffixes (``_full_seq8_sliced``).
    """
    name = bundle.name
    for token in ("_int8_", "_int8"):
        if token in name:
            return name.split(token)[0]
    raise ValueError(f"{name}: not an int8 bundle name")


def load_bundle_inputs(bundle: Path) -> list[np.ndarray]:
    """The bundle's own recorded inputs, in capture order (``in0``, ``in1``, ...)."""
    with np.load(bundle / "inputs.npz") as data:
        keys = sorted(data.files, key=lambda k: int(k[2:]) if k.startswith("in") else 1 << 30)
        return [np.asarray(data[k]) for k in keys]


# --------------------------------------------------------------------------- driver half
def _load_capture_toml(model_dir: Path) -> dict:
    path = model_dir / "capture.toml"
    if not path.is_file():
        return {}
    import tomllib
    return tomllib.loads(path.read_text(encoding="utf-8"))


def capture_python(m2m_root: Path, model: str) -> Path:
    """The interpreter the model's capture runs under.

    ``capture.toml`` ``venv`` when declared (several workloads point at the shared repo venv);
    otherwise the per-model ``.venv``, and failing that the repo-level one. The last fallback
    matters because a workload with no ``capture.toml`` at all (``small_llama``) was captured
    under the shared venv, and resolving only the per-model default reports it as unbuildable.
    """
    model_dir = m2m_root / "workloads" / model
    value = Path(str(_load_capture_toml(model_dir).get("venv", ".venv")))
    venv = value if value.is_absolute() else (model_dir / value).resolve()
    python = venv / "bin" / "python"
    if not python.is_file():
        shared = m2m_root / ".venv" / "bin" / "python"
        if shared.is_file():
            return shared
    return python


def m2m_root() -> Path:
    from merlin.baselines.bundle import model2mlir_root
    return model2mlir_root()


def int8_bundles() -> list[Path]:
    from merlin.common.artifacts import recaptures_dir
    root = recaptures_dir()
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir()
                  if d.is_dir() and "_int8" in d.name and (d / "weights.safetensors").is_file())


def resolve_bundle(name: str) -> Path:
    from merlin.common.artifacts import recaptures_dir
    path = Path(name)
    return path if path.is_dir() else recaptures_dir() / name


def capture_environment(root: Path, model: str,
                        overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for the in-venv worker: the process env, then ``capture.toml`` ``[env]``,
    then explicit overrides (an empty override value UNSETS the variable).

    The override layer is not a convenience. ``capture.toml`` carries the SMOKE configuration
    for models whose full capture is expensive (gemma2_2b pins ``M2M_GEMMA_LAYERS=2`` while the
    ``_full`` bundle was captured with the variable unset, i.e. all 26 layers), so applying the
    file blindly would build a different model than the bundle — which the consistency gate
    would correctly refuse, with a message pointing at the weights instead of at the env.
    """
    env = dict(os.environ)
    env.update({k: str(v) for k, v in (_load_capture_toml(root / "workloads" / model)
                                       .get("env", {}) or {}).items()})
    for key, value in (overrides or {}).items():
        if value == "":
            env.pop(key, None)
        else:
            env[key] = value
    return env


def generate(bundle: Path, *, out_name: str = DEFAULT_OUT_NAME, force: bool = False,
             timeout: int = 7200, env_overrides: dict[str, str] | None = None,
             scheme: str = ACT_QUANT_SCHEME) -> tuple[bool, str]:
    """Drive the in-venv worker for one bundle. Returns ``(ok, message)``."""
    target = bundle / out_name
    if target.is_file() and not force:
        return True, "already present (use --force to regenerate)"
    model = bundle_model_name(bundle)
    root = m2m_root()
    python = capture_python(root, model)
    if not python.is_file():
        return False, f"capture interpreter is absent: {python}"
    env = capture_environment(root, model, env_overrides)
    env.setdefault("TMPDIR", "/scratch/agustin/tmp" if Path("/scratch/agustin/tmp").is_dir()
                   else env.get("TMPDIR", "/tmp"))
    cmd = [str(python), str(Path(__file__).resolve()), "--_inner", str(bundle),
           "--model", model, "--m2m-root", str(root), "--out-name", out_name,
           "--scheme", scheme]
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith("__INDEPENDENT_GOLDEN__ "):
            payload = json.loads(line[len("__INDEPENDENT_GOLDEN__ "):])
    if payload is None:
        tail = (proc.stdout[-1500:] + "\n" + proc.stderr[-1500:]).strip()
        return False, f"worker produced no result (rc={proc.returncode}):\n{tail}"
    if not payload.get("ok"):
        return False, f"{payload.get('reason', 'refused')}: {json.dumps(payload.get('weights', {}))}"
    return True, (f"wrote {payload['path']} shape={payload['shape']} in {time.time() - t0:.0f}s; "
                  f"quantized weights matched {payload['weights']['n_quantized']} tensors; "
                  f"cos vs fp32 golden {payload['cos_vs_fp32_golden']:.6f}"
                  + (f", cos vs self-generated golden_w8a8 {payload['cos_vs_self_golden']:.6f}"
                     if payload.get("cos_vs_self_golden") is not None else ""))


# --------------------------------------------------------------------------- in-venv worker
def _inner(bundle: Path, model: str, root: Path, out_name: str, scheme: str) -> int:
    """Runs INSIDE the model's capture venv. Emits one ``__INDEPENDENT_GOLDEN__`` JSON line."""
    import torch

    def emit(payload: dict) -> None:
        print("__INDEPENDENT_GOLDEN__ " + json.dumps(payload), flush=True)

    workloads = root / "workloads"
    sys.path.insert(0, str(workloads))
    sys.path.insert(0, str(workloads / model))
    import m2m
    from m2m.capture.torchao_pipeline import QuantizationConfig
    from loader import get_model_and_inputs                             # type: ignore

    # Same seeding as capture_consistent.py: the instance must be the one the bundle came from.
    torch.manual_seed(0)
    np.random.seed(0)
    mdl, seeded_inputs = get_model_and_inputs()
    if callable(getattr(mdl, "write_bundle", None)):
        # capture_consistent.py routes these to the loader's own multi-program writer (a causal
        # prefill+decode SESSION, not one forward), so there is no single tensor to reference and
        # no single quantized instance to compare. Say so instead of crashing on `.eval()`.
        emit({"ok": False, "reason": f"{model} is a multi-program (session) capture; this tool "
                                     "references a single forward"})
        return 5
    mdl.eval()
    with torch.no_grad():
        # capture_consistent.py perturbs exactly-zero parameters (adaLN-zero heads) unless the
        # loader declares itself paper_ready. Mirror it, or the weights cannot match.
        if not bool(getattr(mdl, "paper_ready", False)):
            for parameter in mdl.parameters():
                if float(parameter.detach().abs().max()) == 0.0:
                    parameter.copy_(torch.randn_like(parameter) * 0.02)

    # Use the BUNDLE's own recorded inputs; a re-seeded input is only equal by luck.
    recorded = load_bundle_inputs(bundle)
    seeded = [t.detach().cpu().numpy() for t in seeded_inputs]
    if len(recorded) != len(seeded):
        emit({"ok": False, "reason": f"bundle has {len(recorded)} inputs, loader gives {len(seeded)}"})
        return 3
    inputs = tuple(torch.from_numpy(np.array(a)).to(t.dtype)
                   for a, t in zip(recorded, seeded_inputs))
    seed_reproduces_inputs = all(a.shape == b.shape and np.array_equal(a, b)
                                 for a, b in zip(recorded, seeded))

    # The freshly quantized weights are a scratch copy (up to 5 GB for a 2B model): written only
    # so the consistency gate has something to compare, and discarded once it has run.
    work = Path(tempfile.mkdtemp(prefix=f"indep_w8a8_{bundle.name}_",
                                 dir=os.environ.get("TMPDIR") or None))
    try:
        weights_path = str(work / "weights.safetensors")
        result = m2m.convert(mdl, inputs, backend="fx_importer",
                             quantization=QuantizationConfig(scheme=scheme),
                             level="linalg-on-tensors", weights_path=weights_path)
        if not result.ok:
            emit({"ok": False, "reason": f"m2m.convert failed under scheme {scheme}"})
            return 2

        # ---- CONSISTENCY GATE. No write unless the quantized instance carries the same quantized
        # weights the bundle ships. See the module docstring: this refusal is the artifact.
        report = quantized_weight_diff(read_safetensors(bundle / "weights.safetensors"),
                                       read_safetensors(Path(weights_path)))
        report["source"] = "weights.safetensors"
        if not report["n_quantized"]:
            # The safetensors carry no integer tensor to compare: this capture externalized the
            # integer data as qinner:: in extra.npz instead. Compare THAT container rather than
            # reporting an empty intersection, which would be an unmeasured comparison. Read only
            # in this branch — extra.npz is a gigabyte on the LLM bundles, which do carry their
            # integer weights in the safetensors.
            report = quantized_weight_diff(bundle_quantized_parameters(bundle),
                                           flatten_quantized_parameters(mdl))
            report["source"] = "extra.npz qinner::"
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if not report["ok"]:
        emit({"ok": False, "weights": report,
              "reason": ("no integer weight tensor is shared with the bundle (unmeasured, not "
                         "agreement)" if not report["n_quantized"]
                         else f"{report['n_mismatched']} quantized weight tensor(s) differ "
                              f"from the bundle's")})
        return 3

    with torch.no_grad():
        raw = mdl(*inputs)
    tensor = raw[0] if isinstance(raw, (tuple, list)) else raw
    arr = tensor.detach().float().cpu().numpy()

    fp32_golden = np.load(bundle / "golden.npy")
    if arr.shape != fp32_golden.shape:
        emit({"ok": False, "weights": report,
              "reason": f"shape {arr.shape} != fp32 golden {fp32_golden.shape}"})
        return 4

    target = bundle / out_name
    np.save(target, arr)
    self_golden = bundle / "golden_w8a8.npy"
    cos_self = (cosine(arr, np.load(self_golden)) if self_golden.is_file() else None)
    provenance = {
        "producer": "build_tools/scripts/make_w8a8_independent_golden.py",
        "computed_by": f"torchao {scheme} in torch eager (model2MLIR capture venv)",
        "scheme": scheme,
        "independent_of_merlin": True,
        "decides": (f"whether merlin's arithmetic under {scheme} is correct"
                    if scheme == ACT_QUANT_SCHEME
                    else f"the torch-eager reference for {scheme} (NOT the W8A8 tier)"),
        "consistency_gate": "quantized weights equal the bundle's bit-for-bit",
        "weights": report,
        "inputs": "bundle inputs.npz (recorded), not re-seeded",
        "seed_reproduces_bundle_inputs": bool(seed_reproduces_inputs),
        "bundle": bundle.name,
        "model": model,
        "torch": torch.__version__,
        "cos_vs_fp32_golden": cosine(arr, fp32_golden),
        "cos_vs_self_generated_w8a8_golden": cos_self,
        "created": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
    }
    (bundle / (out_name + ".provenance.json")).write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    emit({"ok": True, "path": str(target), "shape": list(arr.shape), "weights": report,
          "cos_vs_fp32_golden": provenance["cos_vs_fp32_golden"],
          "cos_vs_self_golden": cos_self,
          "seed_reproduces_bundle_inputs": bool(seed_reproduces_inputs)})
    return 0


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundles", nargs="*", help="bundle names or paths under recaptures/")
    parser.add_argument("--list", action="store_true",
                        help="report which int8 bundles have an independent reference")
    parser.add_argument("--force", action="store_true", help="regenerate even if one exists")
    parser.add_argument("--out-name", default=DEFAULT_OUT_NAME,
                        help=f"file written beside golden_w8a8.npy (default {DEFAULT_OUT_NAME})")
    parser.add_argument("--timeout", type=int, default=7200, help="per-bundle worker timeout (s)")
    parser.add_argument("--env", action="append", default=[], metavar="NAME=VALUE",
                        help="override a capture.toml [env] entry; NAME= (empty) unsets it")
    parser.add_argument("--_inner", dest="inner", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--m2m-root", dest="m2m_root", help=argparse.SUPPRESS)
    parser.add_argument("--scheme", default=ACT_QUANT_SCHEME,
                        help="torchao scheme to reference against (default: the W8A8 one). Pass "
                             "the bundle's own weight-only scheme with a different --out-name to "
                             "re-derive the fp32 golden independently.")
    args = parser.parse_args(argv)

    if args.inner:
        return _inner(Path(args.bundles[0]), args.model, Path(args.m2m_root),
                      args.out_name, args.scheme)

    if args.list:
        for bundle in int8_bundles():
            have = (bundle / args.out_name).is_file()
            print(f"{'HAVE' if have else '----'}  {bundle.name}")
        return 0

    if not args.bundles:
        parser.error("give at least one bundle name (or --list)")

    overrides: dict[str, str] = {}
    for item in args.env:
        key, sep, value = item.partition("=")
        if not sep:
            parser.error(f"--env expects NAME=VALUE (got {item!r})")
        overrides[key] = value

    failures = 0
    for name in args.bundles:
        bundle = resolve_bundle(name)
        if not bundle.is_dir():
            print(f"!! {name}: no such bundle")
            failures += 1
            continue
        try:
            ok, message = generate(bundle, out_name=args.out_name, force=args.force,
                                   timeout=args.timeout, env_overrides=overrides,
                                   scheme=args.scheme)
        except Exception as exc:                                        # noqa: BLE001
            ok, message = False, f"{type(exc).__name__}: {exc}"
        print(f"{'OK ' if ok else '!! '}{bundle.name}: {message}")
        failures += 0 if ok else 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
