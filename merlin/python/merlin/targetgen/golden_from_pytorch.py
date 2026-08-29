"""Materialize a capsule's INDEPENDENT float golden from its own declared PyTorch reference.

Why this exists
---------------
:func:`merlin.targetgen.capsule_golden.golden` has two paths: an *independent float golden* READ from
``golden.yaml``, and a recompute on the integer :class:`~merlin.runtime.tensor.Tensor` engine. The
integer engine only implements the contraction family (matmul/conv2d/attention_qk/attention_pv/
resident_reuse), so a FLOAT capsule whose op is a normalization / activation / attention composition
has no gradeable golden at all — it raises ``golden: unsupported operation`` and the capsule dies
``RUNNER_CRASH`` before the submission under test is ever executed. That is a defect in the ORACLE,
not in the submission, and it silently caps a corpus's measurable score.

This module closes that gap the way the contract already anticipates: a capsule that declares a
``pytorch_ref`` (``{op, dtype, loader}``) carries its own self-contained, seeded reference definition,
and the loader's torch-eager result IS the reference answer. We run it, then write the ``golden.yaml``
the grader reads.

Target-agnostic by construction
-------------------------------
Nothing here knows a target, an op list, or a capsule name. The op is whatever the capsule declares;
the operands come from the capsule's OWN loader; the shapes/dtypes are validated against the capsule's
OWN ``inputs`` block. Adding an op or a target needs no edit here. Two properties are deliberate:

* **Independent.** The golden comes from torch, NOT from Merlin's own engine, so grading a Merlin-
  generated compiler against it is a genuine cross-check rather than a tautology.
* **Fail closed.** Any disagreement between the loader and the declared contract (operand count,
  shape, an unmapped multi-output, a missing loader) raises instead of writing a plausible-looking
  answer key. A wrong golden is worse than an absent one: it converts a correct submission into a
  numeric failure, or blesses an incorrect one.

The emitted ``golden.yaml`` is an ANSWER SURFACE: it is gitignored (``**/golden.yaml``) and masked
from the agent by the sandbox. Never track it, never place it inside a bundle-visible path.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import yaml

#: capsule ``dtype`` spelling -> torch dtype attribute name. The capsule contract spells float widths
#: ``fp16``/``bf16``/``fp32`` and MLIR spells them ``f16``/``f32``; accept both rather than assuming one.
_DTYPES = {
    "fp16": "float16", "f16": "float16", "half": "float16",
    "bf16": "bfloat16",
    "fp32": "float32", "f32": "float32", "float": "float32",
    "fp64": "float64", "f64": "float64",
    # INTEGER operands. A capsule that takes token ids (an embedding lookup, a gather) declares an
    # integer input, and without these it failed closed with "unmapped dtype 'i64'" and got no golden
    # at all -- so the capsule could not be graded, which reads as a missing answer key rather than as
    # the generator lacking a dtype.
    "i8": "int8", "int8": "int8", "u8": "uint8", "uint8": "uint8",
    "i16": "int16", "int16": "int16",
    "i32": "int32", "int32": "int32",
    "i64": "int64", "int64": "int64",
}

#: Torch dtypes that INDEX rather than carry a value. They must never be widened to the accumulation
#: dtype: `nn.Embedding` and `gather` require integer indices and raise on a float tensor, and a
#: token id has no meaningful reduced-precision rounding to model.
_INDEX_DTYPES = frozenset({"int8", "uint8", "int16", "int32", "int64"})


class GoldenGenError(RuntimeError):
    """A capsule's declared reference disagrees with its contract (fail-closed; no golden written)."""


def _torch():
    try:
        import torch
    except ModuleNotFoundError as e:  # pragma: no cover - environment-dependent
        raise GoldenGenError(
            "torch is required to materialize a pytorch_ref golden; run this under an interpreter "
            "that has torch installed") from e
    return torch


def _load_loader(capsule_dir: Path, loader_name: str):
    """Import the capsule's loader module by PATH (it is capsule-local data, not an installed module)."""
    path = capsule_dir / loader_name
    if not path.is_file():
        raise GoldenGenError(f"declared pytorch_ref loader is missing: {path}")
    # A unique module name per capsule dir keeps two capsules' loaders from colliding in sys.modules.
    mod_name = f"_capsule_ref_{hashlib.sha1(str(path).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise GoldenGenError(f"cannot import capsule loader: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "get_model_and_inputs"):
        raise GoldenGenError(f"{path} defines no get_model_and_inputs()")
    return mod


def _as_tuple(x: Any) -> tuple:
    return tuple(x) if isinstance(x, (tuple, list)) else (x,)


def _shape_of(t) -> list[int]:
    return [int(d) for d in tuple(t.shape)]


def _nested(t) -> Any:
    """Row-major nested lists of python floats (json/yaml-safe; float64 so no width is lost on write)."""
    return t.detach().to(dtype=_torch().float64).cpu().tolist()


def _flat(x: Any) -> list:
    if not isinstance(x, (list, tuple)):
        return [x]
    out: list = []
    for e in x:
        out.extend(_flat(e)) if isinstance(e, (list, tuple)) else out.append(e)
    return out


def _out_names(capsule: dict) -> list[str]:
    """Declared output names, in order. Single-output capsules spell it ``attributes.out``; a
    multi-output composition spells an ``outs`` list. Anything else is an unmapped contract we refuse
    to guess at."""
    attrs = (capsule.get("operation") or {}).get("attributes") or {}
    outs = attrs.get("outs")
    if isinstance(outs, (list, tuple)) and outs:
        return [str(o) for o in outs]
    return [str(attrs.get("out", "Y0"))]


def build_golden(capsule_dir: str | Path) -> dict:
    """Run the capsule's declared PyTorch reference and return the ``golden.yaml`` document.

    Raises :class:`GoldenGenError` (never returns a guess) when the loader and the capsule contract
    disagree, so a bad answer key cannot silently enter the corpus.
    """
    torch = _torch()
    cdir = Path(capsule_dir)
    capsule = yaml.safe_load((cdir / "capsule.yaml").read_text(encoding="utf-8"))

    ref = capsule.get("pytorch_ref") or {}
    loader_name = ref.get("loader")
    if not loader_name:
        raise GoldenGenError(f"{cdir.name}: capsule declares no pytorch_ref.loader")

    declared = list(capsule.get("inputs") or [])
    mod = _load_loader(cdir, loader_name)
    model, raw_inputs = mod.get_model_and_inputs()
    operands = _as_tuple(raw_inputs)

    if len(operands) != len(declared):
        raise GoldenGenError(
            f"{cdir.name}: loader returned {len(operands)} operand(s) but the capsule declares "
            f"{len(declared)} input(s) — refusing to guess the mapping")

    # Bind loader operands to declared leaves POSITIONALLY, but only after every shape agrees. A
    # shape match across the whole list is what makes the positional binding safe to rely on; a
    # mismatch means the loader and the contract have drifted and the golden would be meaningless.
    bound: list[tuple[dict, Any]] = []
    for spec, operand in zip(declared, operands):
        want = [int(d) for d in (spec.get("shape") or [])]
        got = _shape_of(operand)
        if want and want != got:
            raise GoldenGenError(
                f"{cdir.name}: input {spec.get('name')!r} declares shape {want} but the loader "
                f"produced {got} — contract/loader drift, no golden written")
        bound.append((spec, operand))

    # Round each operand to the width the DEVICE will actually see, so the golden is the reference
    # answer for exactly those operands (the remaining device-vs-reference gap is accumulation order,
    # which the capsule's own tolerance policy is there to absorb).
    cast_ops = []
    for spec, operand in bound:
        dt = _DTYPES.get(str(spec.get("dtype", "fp32")).lower())
        if dt is None:
            raise GoldenGenError(f"{cdir.name}: input {spec.get('name')!r} has unmapped dtype "
                                 f"{spec.get('dtype')!r}")
        cast_ops.append(operand.to(dtype=getattr(torch, dt)))

    # Evaluate the reference in the capsule's compare width (default f32): the operands are already
    # device-width, and computing the REFERENCE at reduced precision would bake one particular
    # accumulation order into the answer key.
    acc_name = _DTYPES.get(str(((capsule.get("numeric_policy") or {}).get("dtype")) or "fp32").lower(),
                           "float32")
    acc = getattr(torch, acc_name)
    with torch.no_grad():
        # Integer operands are passed THROUGH: see _INDEX_DTYPES. Casting them to the accumulation
        # width would hand an embedding lookup a float index tensor and raise inside the reference.
        result = model(*[o if o.dtype.is_floating_point is False else o.to(dtype=acc)
                         for o in cast_ops])
    produced = _as_tuple(result)

    names = _out_names(capsule)
    if len(produced) != len(names):
        raise GoldenGenError(
            f"{cdir.name}: reference produced {len(produced)} output(s) but the capsule declares "
            f"{len(names)} ({names}) — declare 'outs' to map a multi-output composition")

    inputs_prov: dict[str, dict] = {}
    for (spec, _), cast in zip(bound, cast_ops):
        inputs_prov[str(spec["name"])] = {
            "shape": _shape_of(cast),
            "dtype": str(spec.get("dtype")),
            "decoded": _flat(_nested(cast)),
        }

    loader_sha = hashlib.sha256((cdir / loader_name).read_bytes()).hexdigest()
    return {
        # Any value other than 'merlin_tensor_int' routes golden() down the INDEPENDENT float path.
        "golden_source": f"torch_eager_{acc_name}",
        "oracle_provenance": {
            "generator": "merlin.targetgen.golden_from_pytorch",
            "reference": {"loader": loader_name, "loader_sha256": loader_sha,
                          "declared_op": (ref.get("op") or (capsule.get("operation") or {}).get("op")),
                          # str(): torch exposes a TorchVersion (str subclass) that yaml refuses to
                          # represent, which fails the WRITE after a correct compute.
                          "torch_version": str(torch.__version__),
                          "eval_dtype": acc_name},
            "inputs": inputs_prov,
        },
        "outputs": {name: _nested(t.to(dtype=acc)) for name, t in zip(names, produced)},
    }


def write_golden(capsule_dir: str | Path, *, overwrite: bool = False) -> Path:
    """Materialize ``<capsule_dir>/golden.yaml``. Refuses to clobber unless ``overwrite``."""
    cdir = Path(capsule_dir)
    dest = cdir / "golden.yaml"
    if dest.exists() and not overwrite:
        raise GoldenGenError(f"{dest} already exists (pass overwrite=True to replace)")
    doc = build_golden(cdir)
    dest.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return dest


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="materialize independent float goldens from capsules' declared pytorch_ref loaders")
    ap.add_argument("--corpus", required=True, help="corpus root to walk for capsule.yaml")
    ap.add_argument("--capsule", action="append", default=[],
                    help="restrict to this capsule name (repeatable)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="compute + validate, write nothing")
    a = ap.parse_args(argv)

    dirs = sorted(p.parent for p in Path(a.corpus).rglob("capsule.yaml"))
    if a.capsule:
        keep = set(a.capsule)
        dirs = [d for d in dirs if d.name in keep]

    wrote, skipped, failed = [], [], []
    for d in dirs:
        capsule = yaml.safe_load((d / "capsule.yaml").read_text(encoding="utf-8"))
        if not ((capsule.get("pytorch_ref") or {}).get("loader")):
            skipped.append((d.name, "no pytorch_ref.loader"))
            continue
        try:
            if a.dry_run:
                # Serialize too: a document that computes correctly but cannot be represented in yaml
                # would otherwise pass --dry-run and fail only on the real write.
                yaml.safe_dump(build_golden(d), sort_keys=False)
                wrote.append(d.name)
            else:
                write_golden(d, overwrite=a.overwrite)
                wrote.append(d.name)
        except GoldenGenError as e:
            failed.append((d.name, str(e)))
        except Exception as e:  # noqa: BLE001 - report, never half-write
            failed.append((d.name, f"{type(e).__name__}: {e}"))

    verb = "validated" if a.dry_run else "wrote"
    print(f"{verb}: {len(wrote)}   skipped: {len(skipped)}   failed: {len(failed)}")
    for n in wrote:
        print(f"  [ok]      {n}")
    for n, why in skipped:
        print(f"  [skip]    {n}: {why}")
    for n, why in failed:
        print(f"  [FAIL]    {n}: {why}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
