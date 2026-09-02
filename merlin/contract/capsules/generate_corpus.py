#!/usr/bin/env python3
"""Unified capsule-corpus generator — ONE generator for every target.

Replaces the two forked generators (this file's gemmini/integer predecessor + ``atlas/generate_atlas_corpus.py``
atlas/float). For each target it loads the declarative ``profiles/<target>.yaml`` (the target-agnostic test
DEFINITION — op + shapes-in-tiles + epilogue, plus the numeric datapath), derives the per-target binding from
the target's descriptor via :mod:`merlin.targetgen.corpus_spec` (dtypes, tile dim, instruction classes, oracle
tiers — nothing hand-set per target), builds each capsule, computes its golden with the regime's engine
(integer = the :mod:`capsule_golden` recompute; float = the external ``specir`` fp8/bf16 refmodel), and writes
the 5-file capsule dir into the target's own corpus root (``Path(te.capsule_corpus).parent`` — gemmini at the
contract root, atlas under ``atlas/``). Only capsules named in a profile are (over)written; hand-authored
capsules (e.g. gemmini's movement/conv) are left untouched.

Run:  PYTHONPATH=$SPECIR_ROOT .venv/bin/python \
          merlin/contract/capsules/generate_corpus.py            # all targets with a profile
      ... merlin/contract/capsules/generate_corpus.py --target gemmini
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "merlin" / "python"))

from merlin.common.paths import _dotenv                   # noqa: E402
from merlin.perf import workload_gen as WG                   # noqa: E402
from merlin.perf.profile import TRAITS, derive_profile       # noqa: E402
from merlin.targetgen import capsule_golden as CG            # noqa: E402
from merlin.targetgen import corpus_spec as CS               # noqa: E402
from merlin.targetgen import numeric_falsifiability as NF    # noqa: E402
from merlin.targetgen.target_experiment import load_target_experiment  # noqa: E402

HERE = Path(__file__).resolve().parent
PROFILES = HERE / "profiles"

_PERFORMANCE_FIELDS = frozenset({
    "level", "family", "lever", "comparand", "falsifier", "gate", "regime", "emitter", "cost",
})
_PERFORMANCE_CLAIMS = frozenset({"RECOVERS", "PREDICTS", "DIFFERENTIAL"})
_PERFORMANCE_NESTED_FIELDS = {
    "comparand": frozenset({"kind", "against", "cancels", "demand_equal"}),
    "falsifier": frozenset({"observation", "fires_when", "negative_control"}),
    "gate": frozenset({"traits", "instrument", "capacity", "on_missing"}),
    "regime": frozenset({"separation", "layout"}),
    "emitter": frozenset({"status", "entry", "knobs"}),
    "cost": frozenset({"tier", "runs", "projected_cycles", "basis"}),
}


def _document_digest(document) -> str:
    """Stable digest for a parsed declaration/fact document."""
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_performance_block(block, *, owner: str) -> dict:
    """Validate the claim-bearing contract before a family can be admitted.

    Performance metadata is consumed later than corpus generation, so a partial
    block otherwise succeeds here and turns into an unmeasurable family only
    after an expensive run.  All claim, comparison, falsifier, applicability,
    regime, emitter, and cost fields therefore fail closed at profile load.
    """
    if not isinstance(block, dict):
        raise ValueError(f"{owner}: `performance` must be a mapping")
    missing = sorted(_PERFORMANCE_FIELDS - block.keys())
    if missing:
        raise ValueError(f"{owner}: performance block missing required field(s) {missing}")
    for field in ("level", "family", "lever"):
        if not isinstance(block[field], str) or not block[field].strip():
            raise ValueError(f"{owner}: performance.{field} must be a non-empty string")
    claim = block.get("claim")
    if claim not in _PERFORMANCE_CLAIMS:
        raise ValueError(
            f"{owner}: performance.claim must be one of {sorted(_PERFORMANCE_CLAIMS)}, got {claim!r}")
    gate_value = block.get("gate")
    if isinstance(gate_value, dict) and "requires" in gate_value:
        raise ValueError(
            f"{owner}: performance.gate.requires is not accepted; use canonical `gate.traits`")
    for field, required in _PERFORMANCE_NESTED_FIELDS.items():
        value = block[field]
        if not isinstance(value, dict):
            raise ValueError(f"{owner}: performance.{field} must be a mapping")
        absent = sorted(required - value.keys())
        if absent:
            raise ValueError(
                f"{owner}: performance.{field} missing required field(s) {absent}")
        for key in required - {"knobs"}:
            nested = value[key]
            if (nested is None or (isinstance(nested, str) and not nested.strip())
                    or (isinstance(nested, (list, tuple, dict)) and not nested)):
                raise ValueError(
                    f"{owner}: performance.{field}.{key} must be non-empty")
    gate = block["gate"]
    names = gate["traits"]
    if (not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not name for name in names)):
        raise ValueError(f"{owner}: performance.gate.traits must be a non-empty list of trait names")
    if len(set(names)) != len(names):
        raise ValueError(f"{owner}: performance.gate.traits contains duplicate names {names}")
    unknown = sorted(set(names) - set(TRAITS))
    if unknown:
        raise ValueError(
            f"{owner}: unknown performance trait(s) {unknown}; canonical traits are {list(TRAITS)}")
    if gate["on_missing"] != "skip_with_evidence":
        raise ValueError(
            f"{owner}: performance.gate.on_missing must be 'skip_with_evidence'")
    emitter = block["emitter"]
    if not isinstance(emitter["status"], str) or not emitter["status"]:
        raise ValueError(f"{owner}: performance.emitter.status must be non-empty")
    if not isinstance(emitter["entry"], str) or not emitter["entry"]:
        raise ValueError(f"{owner}: performance.emitter.entry must be non-empty")
    if not isinstance(emitter["knobs"], dict):
        raise ValueError(f"{owner}: performance.emitter.knobs must be a mapping")
    return block


def _comparison_roles(sweep: dict) -> list[str]:
    roles = {str(role) for role in (sweep.get("comparison_roles") or []) if str(role)}
    roles |= {str(group["role"])
              for variant in (sweep.get("variants") or [])
              if isinstance(variant, dict)
              if isinstance((group := variant.get("comparison_group")), dict)
              if group.get("role")}
    return sorted(roles)


def _validate_declared_fit_axes(sweep: dict, *, owner: str) -> None:
    fit_axes = sweep.get("fit_axes") or []
    if (not isinstance(fit_axes, list) or not fit_axes
            or any(not isinstance(axis, str) or not axis for axis in fit_axes)):
        raise ValueError(f"{owner}: fit_axes must be a non-empty list of axis names")
    axes = sweep.get("axes") or {}
    unknown = [axis for axis in fit_axes if axis not in axes]
    if unknown:
        raise ValueError(f"{owner}: fitted axes {unknown} are not declared in axes")
    for axis in fit_axes:
        points = axes[axis]
        if not isinstance(points, list) or len({repr(point) for point in points}) < 2:
            raise ValueError(
                f"{owner}: fitted axis {axis} must declare at least two distinct points")


def _performance_family_record(sweep: dict) -> dict:
    performance = (sweep.get("base") or {}).get("performance") or sweep.get("performance") or {}
    return {
        "family": performance.get("family") or sweep.get("id"),
        "claim": performance.get("claim"),
        "fit_axes": list(sweep.get("fit_axes") or []),
        "comparison_roles": _comparison_roles(sweep),
    }


def _target_local_perf_declarations(profile: dict) -> list[str]:
    """Names of performance entries embedded in a target-owned profile.

    Functional sweeps remain target-owned: they describe operation coverage for
    that target.  Performance families do not.  Their shapes and comparison
    structure must come from the one shared ``_perf.yaml`` template, otherwise
    onboarding a target can quietly fork the experiment it is compared under.
    """
    found: list[str] = []
    for entry in profile.get("capsules") or []:
        if (isinstance(entry, dict)
                and (entry.get("cat") in {"perf", "_perf"} or "performance" in entry)):
            found.append(str(entry.get("name") or "<unnamed capsule>"))
    for sweep in profile.get("sweeps") or []:
        if not isinstance(sweep, dict):
            continue
        base = sweep.get("base") or {}
        variants = sweep.get("variants") or []
        if ((isinstance(base, dict)
             and (base.get("cat") in {"perf", "_perf"} or "performance" in base))
                or any(isinstance(v, dict)
                       and (v.get("cat") in {"perf", "_perf"} or "performance" in v)
                       for v in variants)):
            found.append(str(sweep.get("id") or "<unnamed sweep>"))
    return found


def _merge_shared_perf(profile: dict, *, source: Path) -> None:
    """Merge the sole shared performance template into one target profile."""
    shared_path = PROFILES / "_perf.yaml"
    shared = yaml.safe_load(shared_path.read_text(encoding="utf-8")) or {}
    misplaced = _target_local_perf_declarations(profile)
    if misplaced:
        raise ValueError(
            f"{source} declares target-local performance template entries {misplaced}; "
            f"move them to the shared {shared_path}")
    if shared.get("capsules"):
        raise ValueError(
            f"shared performance template {shared_path} must generate entries through `sweeps`, "
            "not hand-author `capsules`")
    non_perf = [
        str(s.get("id") or "<unnamed sweep>")
        for s in (shared.get("sweeps") or [])
        if not isinstance(s, dict) or (s.get("base") or {}).get("cat") != "_perf"
    ]
    if non_perf:
        raise ValueError(
            f"shared performance template {shared_path} contains non-performance sweeps {non_perf}")
    sweeps = list(shared.get("sweeps") or [])
    blocked = list(shared.get("blocked_unimplemented") or [])
    family_records: list[dict] = []
    seen: set[str] = set()
    for sweep in sweeps:
        sweep_id = str(sweep.get("id") or "").strip()
        if not sweep_id:
            raise ValueError(f"shared performance template {shared_path} has a sweep without an id")
        performance = _validate_performance_block(
            (sweep.get("base") or {}).get("performance"), owner=f"shared sweep {sweep_id}")
        _validate_declared_fit_axes(sweep, owner=f"shared sweep {sweep_id}")
        if performance["family"] != sweep_id:
            raise ValueError(
                f"shared sweep {sweep_id}: performance.family must equal the sweep id")
        if sweep_id in seen:
            raise ValueError(f"shared performance template repeats family {sweep_id!r}")
        seen.add(sweep_id)
        family_records.append(_performance_family_record(sweep))
    for item in blocked:
        if not isinstance(item, dict):
            raise ValueError(f"shared {shared_path}: blocked_unimplemented entries must be mappings")
        family = str(item.get("family") or "").strip()
        if not family or not str(item.get("reason") or "").strip():
            raise ValueError(
                f"shared {shared_path}: blocked_unimplemented needs family and reason")
        performance = _validate_performance_block(
            item.get("performance"), owner=f"blocked performance family {family}")
        if performance["family"] != family:
            raise ValueError(
                f"blocked family {family}: performance.family must equal its family")
        if family in seen:
            raise ValueError(f"shared performance template repeats family {family!r}")
        seen.add(family)
        family_records.append({
            "family": family,
            "claim": performance.get("claim"),
            "fit_axes": list(item.get("fit_axes") or []),
            "comparison_roles": list(item.get("comparison_roles") or []),
        })
    profile["sweeps"] = list(profile.get("sweeps") or []) + sweeps
    try:
        template_path = str(shared_path.relative_to(REPO))
    except ValueError:
        template_path = str(shared_path)
    profile["_performance_template"] = {
        "path": template_path,
        "sha256": _document_digest(shared),
        "families": family_records,
        "blocked_unimplemented": copy.deepcopy(blocked),
    }


def load_profile(target: str, *, include_holdouts: bool = True) -> dict:
    """The target's functional profile plus shared perf and the private holdout sidecar.

    The holdout spec (op + dtype + exact shape) is an answer, not a contract: the tracked profile lives
    inside the ``merlin/contract/`` tree every arm is granted read-only, so a holdout declared there is
    readable by the agent under test. It therefore lives in ``profiles/<target>.hidden.yaml``, which is
    gitignored and masked by :mod:`merlin.targetgen.sandbox.answer_surfaces`. When the sidecar is absent
    -- a public clone, or a sandbox where it is masked -- this returns the public profile unchanged and
    the run simply emits no hidden capsules, which is the correct behaviour rather than an error.

    ``include_holdouts=False`` for any caller whose OUTPUT is public: a published artifact that
    enumerates the holdouts leaks them just as the profile did.
    """
    public = PROFILES / f"{target}.yaml"
    prof = yaml.safe_load(public.read_text(encoding="utf-8")) or {}
    _merge_shared_perf(prof, source=public)
    # SYNTHESIZED ENTRIES, appended after the hand-authored ones. They come from the target's own
    # derived conformance requirement (build_tools/scripts/synth_capsule_corpus.py --write) and carry
    # the cell each was synthesized for in `source_reference`. Appended rather than prepended so
    # `expand_sweeps`' documented declaration-order semantics are untouched; its `seen` set already
    # raises on a duplicate name, and every synthesized name is `SY_`-prefixed, so a collision with a
    # hand-authored capsule is impossible rather than merely unlikely.
    synth = PROFILES / f"{target}.synth.yaml"
    if synth.is_file():
        doc = yaml.safe_load(synth.read_text(encoding="utf-8")) or {}
        extra = list(doc.get("capsules") or ())
        if extra:
            prof["capsules"] = list(prof.get("capsules") or []) + extra
    side = PROFILES / f"{target}.hidden.yaml"
    if include_holdouts and side.is_file():
        held = yaml.safe_load(side.read_text(encoding="utf-8")) or {}
        misplaced = _target_local_perf_declarations(held)
        if misplaced:
            raise ValueError(
                f"{side} declares target-local performance template entries {misplaced}; "
                f"move them to the shared {PROFILES / '_perf.yaml'}")
        prof["capsules"] = list(prof.get("capsules") or []) + list(held.get("capsules") or [])
        # HOLDOUTS ARE GENERATED TOO, and without this line the sidecar's `sweeps:` block is read and
        # silently dropped -- the points expand for the disjointness gate, which reads the sidecar
        # itself, and then never become capsules. Generating them is the point: a hand-authored holdout
        # is written by someone who has just read the public profile, and an audit of this repo found
        # holdouts that were public capsules under another name, scoring memorisation as transfer. A
        # sweep states the OBLIGATION and lets the tile edge compute the points.
        prof["sweeps"] = list(prof.get("sweeps") or []) + list(held.get("sweeps") or [])
    return prof


def profile_targets() -> list[str]:
    """Target profile stems, excluding shared templates and private sidecars."""
    return sorted(
        path.stem for path in PROFILES.glob("*.yaml")
        if not path.name.startswith("_") and not path.name.endswith(".hidden.yaml"))


# ------------------------------------------------------------------------------------------------
# deterministic local-path scrub (tracked-file hygiene — no /tmp, /scratch, /home paths ever ship).
# The m2m capture externalizes weights to a NON-deterministic temp dir and stamps its ABSOLUTE path into
# the captured linalg's ``prov.weights_file`` module attribute; the whole-model writer relativizes it but
# the fused/mapped op writer (in capsule_source, outside this file's edit boundary) does not — so we scrub
# the written capsule dir HERE, at the one place this generator owns. Op capsules ship NO weights file
# (their operands come from golden.yaml provenance), so the attribute is non-load-bearing and is STRIPPED;
# an already-relative value (the whole-model ``capsule.weights.safetensors``) is left untouched. Captured
# whole-model loader SOURCES (capsule.pytorch.py) may also carry upstream local-path env-defaults / docstring
# examples — those local-path tokens are redacted. Deterministic (temp-dir name never survives) + idempotent;
# structural string ops only (no regex).
# ------------------------------------------------------------------------------------------------
_LOCAL_ROOTS = ("/tmp", "/scratch", "/home")
_PATH_TERMINATORS = set("\"'") | set(" \t\r\n),]}>")


def _strip_weights_attr(text: str) -> str:
    """Strip an ABSOLUTE ``prov.weights_file = "<abs>"`` module attribute (with its separating comma); leave
    an already-relative value alone. There is exactly one per captured-linalg module."""
    key = 'prov.weights_file = "'
    i = text.find(key)
    if i == -1:
        return text
    j = text.find('"', i + len(key))
    if j == -1:
        return text
    if "/" not in text[i + len(key):j]:          # already relative (e.g. capsule.weights.safetensors)
        return text
    start, end = i, j + 1
    if text[end:end + 2] == ", ":                # attribute is first: drop trailing ", "
        end += 2
    elif text[start - 2:start] == ", ":          # attribute is later: drop leading ", "
        start -= 2
    return text[:start] + text[end:]


def _redact_local_paths(text: str) -> str:
    """Replace any absolute local-filesystem path token (``/tmp*`` / ``/scratch*`` / ``/home*``, consumed up
    to the next quote / whitespace / bracket) with the stable placeholder ``<path>``. Idempotent (the
    placeholder contains no local root)."""
    for root in _LOCAL_ROOTS:
        while True:
            i = text.find(root)
            if i == -1:
                break
            j = i + len(root)
            while j < len(text) and text[j] not in _PATH_TERMINATORS:
                j += 1
            text = text[:i] + "<path>" + text[j:]
    return text


def _scrub_capsule_dir(d) -> None:
    """Scrub every tracked-shippable text file in a written capsule dir of non-deterministic / local paths:
    strip the absolute ``prov.weights_file`` from ``*.mlir`` and redact local-path tokens from ``*.mlir`` +
    ``*.py``. Only rewrites a file when its content actually changes (keeps regenerations byte-stable)."""
    if d is None:
        return
    for p in sorted(Path(d).iterdir()):
        if p.suffix not in (".mlir", ".py") or not p.is_file():
            continue
        text = p.read_text(encoding="utf-8")
        scrubbed = text
        if p.suffix == ".mlir":
            scrubbed = _strip_weights_attr(scrubbed)
        scrubbed = _redact_local_paths(scrubbed)
        if scrubbed != text:
            p.write_text(scrubbed, encoding="utf-8")


# ------------------------------------------------------------------------------------------------
# float golden engine (generation-time only; needs the external specir refmodel)
# ------------------------------------------------------------------------------------------------
def _specir():
    root = os.environ.get("SPECIR_ROOT") or _dotenv().get("SPECIR_ROOT")
    if root not in sys.path:
        sys.path.insert(0, root)
    from specir.oracle import dtypes as D
    from specir.oracle.refmodel import fp_reduce
    return D, fp_reduce


# specir fp8 format handle per canonical operand dtype token (fail closed if the refmodel lacks it).
_SPECIR_FP8_ATTR = {"fp8_e4m3": "FP8_E4M3", "fp8_e5m2": "FP8_E5M2"}


def _specir_fp8(D, fmt_token: str):
    attr = _SPECIR_FP8_ATTR.get(fmt_token)
    if attr is None or not hasattr(D, attr):
        raise ValueError(f"specir refmodel has no fp8 format for operand dtype {fmt_token!r} "
                         f"(known: {sorted(_SPECIR_FP8_ATTR)})")
    return getattr(D, attr)


def _det_fp8(D, name, shape, salt, fmt_token, d_fp8):
    """Structured, format-DERIVED operand bytes: distinct rows AND columns + asymmetric (so a wrong row
    stride / base offset / transposed load changes the output), spanning the fp8 format's representable
    range. Replaces the old 11-magnitude flat-hash fill (~6 distinct values, ~11/32 distinct rows) that hid
    those bug classes. See merlin.targetgen.corpus_operands."""
    from merlin.targetgen import corpus_operands as CO
    salt_int = sum((i + 1) * ord(c) for i, c in enumerate(f"{salt}|{name}")) or 1
    vals = CO.operand_values(tuple(shape), fmt_token, salt_int)
    raw = [D.encode_float(v, d_fp8) for v in vals]
    # Self-enforcing rigor: fail generation loudly if the ENCODED bytes are not distinct-per-row/col +
    # asymmetric (e.g. a future palette/fill change, or an encode that collapsed distinct values). A weak
    # operand silently hides addressing/stride/transpose bugs — never let a regeneration ship one.
    if len(shape) == 2:
        problems = CO.rigor_findings([float(b) for b in raw], tuple(shape))
        if problems:
            raise AssertionError(f"non-rigorous operand {name}{tuple(shape)}: {problems}")
    return raw, vals


def _operand_decoder(D, fmt, *, flush_subnormals: bool):
    """Decode a raw operand code the way the DATAPATH decodes it, exactly.

    By default that is the format's own exact value. A datapath that admits only NORMAL operands sees
    zero wherever the operand's exponent field is zero, and a reference model that decodes those codes
    to their tiny nonzero value is modelling different hardware — every later add carries the
    difference. Whether the target does that is a measured property of its compute unit, declared in
    its profile's ``datapath`` block (``subnormal_operand_flush``); nothing here assumes it.

    The subnormal test is DERIVED from the format descriptor the refmodel hands back (the exponent
    field, located by the format's own ``mant_bits``/``exp_bits``), so it holds for any exponent /
    mantissa split rather than one hardcoded byte layout.
    """
    if not flush_subnormals:
        return lambda raw: D.decode_float_exact(int(raw), fmt)
    exp_mask = (1 << fmt.exp_bits) - 1

    def decode(raw):
        raw = int(raw)
        if ((raw >> fmt.mant_bits) & exp_mask) == 0:     # exponent field zero => subnormal (or zero)
            return Fraction(0)
        return D.decode_float_exact(raw, fmt)
    return decode


def _float_golden(entry, binding):
    """A capsule's fp8->bf16 golden + input provenance from the specir refmodel (independent of the RTL)."""
    D, fp_reduce = _specir()
    fmt_token = binding.operand_dtype                    # e.g. "fp8_e4m3" — DERIVED, not assumed
    FP8, BF16 = _specir_fp8(D, fmt_token), D.BF16
    dec = _operand_decoder(D, FP8, flush_subnormals=binding.subnormal_operand_flush)
    salt, dim = entry["name"], binding.tile_dim
    prov, outputs = {}, {}

    def reg(name, shape):
        raw, vals = _det_fp8(D, name, shape, salt, fmt_token, FP8)
        prov[name] = {"shape": list(shape), "fp8_raw_hex": [f"0x{r:02x}" for r in raw], "decoded": vals}
        return raw

    def rnd(x):
        return D.round_to_format(x, BF16, "rne")

    def mm(a_raw, ashape, w_raw, wshape):
        m, k = ashape
        _, n = wshape
        out = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                prods = [rnd(dec(a_raw[i * k + p]) * dec(w_raw[p * n + j])) for p in range(k)]
                out[i][j] = fp_reduce(prods, BF16, order="index_sequential", cadence="per_step", rm="rne")
        return out

    def floats(y):
        return [[D.decode_float(v, BF16) for v in row] for row in y]

    op = entry.get("op", "matmul")
    if op in ("matmul", "linear"):
        M = entry.get("M", entry.get("M_tiles", 1) * dim)
        K = entry.get("K", entry.get("K_tiles", 1) * dim)
        N = entry.get("N", entry.get("N_tiles", 1) * dim)
        a = reg(entry.get("lhs", "A0"), (M, K))
        w = reg(entry.get("weight", "W"), (K, N))
        y = mm(a, (M, K), w, (K, N))
        epi = entry.get("epilogue", [])
        if "acc_scale" in epi:
            s = Fraction(entry["acc_scale"]).limit_denominator(1 << 20)
            y = [[rnd(D.decode_float_exact(v, BF16) * s) for v in row] for row in y]
        if "relu" in epi:
            y = [[v if D.decode_float(v, BF16) > 0 else 0 for v in row] for row in y]
        outputs[entry.get("out", "Y0")] = floats(y)
    elif op == "movement":
        M = entry.get("M", entry.get("M_tiles", 1) * dim)
        N = entry.get("N", entry.get("N_tiles", 1) * dim)
        x = reg(entry.get("src", "X"), (M, N))
        outputs[entry.get("out", "Y0")] = floats([[rnd(dec(x[i * N + j]))
                                                   for j in range(N)] for i in range(M)])
    elif op == "resident_reuse":
        K = entry.get("K_tiles", 1) * dim
        N = entry.get("N_tiles", 1) * dim
        w = reg(entry["weight"], (K, N))
        for m in entry["matmuls"]:
            M = m.get("M_tiles", 1) * dim
            a = reg(m["lhs"], (M, K))
            outputs[m["out"]] = floats(mm(a, (M, K), w, (K, N)))
    elif op == "attention_qk":
        M = entry.get("M_tiles", 1) * dim
        Kd = entry.get("K_tiles", 1) * dim
        q = reg(entry.get("q", "Q"), (M, Kd))
        k = reg(entry.get("k", "K"), (M, Kd))
        kt = [0] * (M * Kd)
        for i in range(M):
            for j in range(Kd):
                kt[j * M + i] = k[i * Kd + j]
        outputs[entry.get("out", "Y0")] = floats(mm(q, (M, Kd), kt, (Kd, M)))
    else:
        raise ValueError(f"no float golden for op {op!r}")
    return outputs, prov


# ------------------------------------------------------------------------------------------------
# MX (microscaling block-scaled FP) golden engine — HARDWARE semantics via mlc's mx_ref, NOT specir
# (specir is the atlas fp8 refmodel; MX is a different datapath: 16-deep systolic per-column accumulate
# schedule + one E8M0 scale per 32-element K group). mx_ref is transcribed bit-exactly from the target's
# own reference (radiance-kernels lib/golden/{mx_fp_math.h,mx_golden.cpp}, mirroring the RTL).
# ------------------------------------------------------------------------------------------------
def _mx_ref():
    """Import mlc's ``validate/mx_ref.py`` BY FILE PATH (like the specir import) so we do NOT trigger
    ``mlc/validate/__init__.py`` (which carries concurrent work and heavy imports)."""
    root = os.environ.get("MERLIN_MLC_DIR") or _dotenv().get("MERLIN_MLC_DIR")
    path = Path(root) / "mlc" / "validate" / "mx_ref.py"
    if not path.exists():
        raise FileNotFoundError(f"mx_ref not found at {path} (set MERLIN_MLC_DIR to the mlc modeling root)")
    spec = importlib.util.spec_from_file_location("merlin_mx_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _salt(name: str, tensor: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(f"{name}|{tensor}")) or 1


def _mx_value_codes(mx, fmt_token: str):
    """value(float) -> device code, DERIVED by decoding every code with mx_ref's own decoder (no baked
    table). fp8: 8-bit e4m3 code; fp4: 4-bit e2m1 nibble; fp6: 6-bit e3m2 code."""
    if fmt_token == "fp8_e4m3":
        rng, dec = range(256), mx.fp8_e4m3_decode
    elif fmt_token == "fp4_e2m1":
        rng, dec = range(16), mx.fp4_e2m1_decode
    elif fmt_token == "fp6_e3m2":
        rng, dec = range(64), mx.fp6_e3m2_decode
    else:
        raise ValueError(f"no MX code table for {fmt_token!r}")
    table: dict[float, int] = {}
    for c in rng:
        v = dec(c)
        if v == v and abs(v) != float("inf"):        # finite; keep the FIRST (lowest) code per value
            table.setdefault(float(v), c)
    return table


def _mx_golden(entry, binding):
    """MX matmul golden (bf16 output) + provenance, computed by mx_ref in hardware semantics. Operands are
    format-derived + rigor-gated; the E8M0 block-scale streams are rigor-gated too (a mis-indexed per-lane
    scale must change the output)."""
    from merlin.runtime.fp8_formats import canonical_float, e8m0_decode
    from merlin.targetgen import corpus_operands as CO
    mx = _mx_ref()
    tok = canonical_float(binding.operand_dtype)          # fp8_e4m3 / fp6_e3m2 / fp4_e2m1
    op = entry.get("op", "matmul")
    if op not in ("matmul", "linear"):
        raise ValueError(f"MX regime supports matmul/linear only (got op {op!r} in {entry['name']!r})")
    dim = binding.tile_dim
    M = entry.get("M", entry.get("M_tiles", 1) * dim)
    K = entry.get("K", entry.get("K_tiles", 1) * dim)
    N = entry.get("N", entry.get("N_tiles", 1) * dim)
    if tok == "fp8_e4m3":
        fmt, max_alpha, G = mx.FMT_FP8, None, 0
    elif tok == "fp4_e2m1":
        fmt, max_alpha, G = mx.FMT_FP4, None, 0
    elif tok == "fp6_e3m2":
        fmt, max_alpha, G = mx.FMT_FP6, 16, 5             # single 16-entry LUT (fp6 is LUT-indexed)
    else:
        raise ValueError(f"unsupported MX operand dtype {tok!r}")
    codes = _mx_value_codes(mx, tok)
    lhs, weight, out = entry.get("lhs", "A0"), entry.get("weight", "W"), entry.get("out", "Y0")

    def synth(name, shape):
        # MX operands are kept small (|v| <= 4): a wide E8M0 block scale over a long-K bf16 accumulate would
        # otherwise saturate to inf (a golden any broken kernel matches). See rand_fp8 in lib/golden.
        vals = CO.operand_values(shape, tok, _salt(entry["name"], name), max_alphabet=max_alpha, mag_cap=4.0)
        problems = CO.rigor_findings(vals, shape)
        if problems:
            raise AssertionError(f"non-rigorous MX operand {name}{shape}: {problems}")
        return np.array(vals, dtype=np.float64).reshape(shape)

    A = synth(lhs, (M, K))
    W = synth(weight, (K, N))

    def enc(v):
        c = codes.get(float(np.float32(v)))
        if c is None:                                     # exactly-representable palette -> exact hit expected
            raise AssertionError(f"MX value {v!r} not exactly representable in {tok}")
        return c

    A_codes = np.vectorize(enc)(A).astype(np.uint8)
    B_codes = np.vectorize(enc)(W).astype(np.uint8)
    # Same partial-block refusal as _mx_requant_blocks: one E8M0 scale per WHOLE group, so a K with a
    # remainder would emit scale streams covering only K - (K % GROUP) elements while the operand codes
    # cover all K. The mismatch is silent -- the scales simply stop early.
    if K % mx.GROUP:
        raise ValueError(
            f"MX golden needs K to be a whole multiple of the {mx.GROUP}-element block-scale group; got "
            f"K={K} for capsule {entry.get('name')!r} ({K % mx.GROUP} element(s) in a partial final "
            f"group). One E8M0 scale is emitted per whole group, so the scale stream would cover only "
            f"{mx.GROUP * (K // mx.GROUP)} of {K} K elements.")
    GK = K // mx.GROUP
    SA = np.array(CO.e8m0_scale_codes((GK, M), _salt(entry["name"], "SA")), dtype=np.uint8)
    SB = np.array(CO.e8m0_scale_codes((GK, N), _salt(entry["name"], "SB")), dtype=np.uint8)
    for nm, sc in (("SA", SA), ("SB", SB)):
        prob = CO.scale_rigor_findings(sc.tolist())
        if prob:
            raise AssertionError(f"non-rigorous E8M0 scale stream {nm}{sc.shape}: {prob}")

    lutA = lutB = None
    if fmt == mx.FMT_FP8:
        Ab, Bb = A_codes, B_codes
    else:
        if fmt == mx.FMT_FP6:                             # nibbles index a shared 16-entry LUT of e3m2 codes
            lut = np.array(sorted({int(c) for c in A_codes.reshape(-1)} |
                                  {int(c) for c in B_codes.reshape(-1)}), dtype=np.uint8)
            assert lut.size <= 16, f"fp6 LUT overflow ({lut.size} > 16)"
            lut = np.pad(lut, (0, 16 - lut.size))[:16]
            idx = {int(v): i for i, v in enumerate(lut)}
            A_nib = np.vectorize(lambda c: idx[int(c)])(A_codes).astype(np.uint8)
            B_nib = np.vectorize(lambda c: idx[int(c)])(B_codes).astype(np.uint8)
            # mx_ref indexes the LUT as ``L[(row_or_col >> G) * 16 + nib]`` — ONE 16-entry block per
            # ``1<<G`` rows (A) / cols (B). Supply exactly that many blocks (a single global palette shared
            # by all groups is replicated: every block is identical, so ``(g)*16 + nib`` always resolves to
            # lut[nib]). Prior code shipped a lone block, so any fp6 capsule with M or N > 1<<G (e.g. N=64)
            # indexed past it and crashed.
            grp = 1 << G
            nblk_A = (A_codes.shape[0] + grp - 1) // grp      # blocks along A rows (M)
            nblk_B = (B_codes.shape[1] + grp - 1) // grp      # blocks along B cols (N)
            lutA = np.tile(lut.reshape(1, 16), (nblk_A, 1))
            lutB = np.tile(lut.reshape(1, 16), (nblk_B, 1))
        else:
            A_nib, B_nib = A_codes, B_codes               # fp4 nibble == code
        Ab = ((A_nib[1::2, :] << 4) | (A_nib[0::2, :] & 0xF)).astype(np.uint8)     # pack along M
        Bb = ((B_nib[:, 1::2] << 4) | (B_nib[:, 0::2] & 0xF)).astype(np.uint8)     # pack along N

    C = mx.mx_matmul(Ab, Bb, SA, SB, M, N, K, fmt=fmt, lutA=lutA, lutB=lutB, G=G)
    y = [[float(mx.bf16_to_f32(int(C[i, j]))) for j in range(N)] for i in range(M)]
    prov = {
        lhs: {"shape": [M, K], "decoded": A.reshape(-1).tolist()},
        weight: {"shape": [K, N], "decoded": W.reshape(-1).tolist()},
        "SA_e8m0_codes": SA.tolist(), "SB_e8m0_codes": SB.tolist(),
        # The SAME scales again, keyed by the operand names the capsule DECLARES, as ordinary per-tensor
        # specs. The two lists above are non-tensor provenance that `canonical_input_raws` skips by
        # design, so before this the scales reached the reference kernel (which bakes them) and no one
        # else -- a submitted backend was handed block-scaled element bytes and no scales, which is half
        # a number. Recorded additively: the oracle keeps reading the lists above.
        f"{lhs}_scale": {"shape": [GK, M], "decoded": SA.reshape(-1).tolist(),
                         "note": "E8M0 exponent codes, one per block of K elements per lhs row"},
        f"{weight}_scale": {"shape": [GK, N], "decoded": SB.reshape(-1).tolist(),
                            "note": "E8M0 exponent codes, one per block of K elements per weight column"},
        "scale_example": {"SA[0][0]": int(SA[0, 0]), "as_scale": e8m0_decode(int(SA[0, 0]))},
        # RAW device operand bytes exactly as mx_ref consumed them (fp8: one byte/elt; fp4/fp6: packed) —
        # the ``decoded`` floats above lose precision through YAML, so a bit-exact grade re-runs the MX
        # datapath oracle over THESE codes, not the decoded values. fmt/dims/LUTs ride along so the grade
        # is self-contained and reproduces the golden exactly.
        "operand_codes": {
            "lhs": lhs, "weight": weight, "fmt": tok, "M": M, "N": N, "K": K, "G": G,
            "A_bytes": Ab.reshape(-1).tolist(), "A_shape": list(Ab.shape),
            "B_bytes": Bb.reshape(-1).tolist(), "B_shape": list(Bb.shape),
            "lutA": lutA.tolist() if lutA is not None else None,
            "lutB": lutB.tolist() if lutB is not None else None,
        },
    }
    return {out: y}, prov


# ------------------------------------------------------------------------------------------------
# MX FUSED FLASH-ATTENTION golden — COMPOSED from the SAME validated mx_ref engine used above:
#   S = mx_matmul(Q, K^T)   (block-scaled E8M0, bf16)  -> scaled by 1/sqrt(head) [+ optional soft-cap]
#   P = bf16 row-softmax(S)                             (numpy, bf16-rounded)
#   O = mx_matmul(P_requant, V)  (block-scaled E8M0, bf16)
# The intermediate P is requantized to the MX code space exactly as the MX PE does before the second GEMM:
# a per-(K-group,row) E8M0 scale brings each block to O(1) mantissas (the mx_ref accumulator carries only
# 4-bit column exponents, so DECODED codes must stay small and the E8M0 scale carry the magnitude — same
# convention as the synthesized Q/K/V operands), then each element rounds to the nearest representable value
# in that format's palette. NOTHING is fabricated: both matmuls are the mlc mx_ref hardware datapath (same
# codec as the R6/R7 fp6/fp4 tiles: fp8 = one byte/code, fp4 = e2m1 nibble, fp6 = e3m2 nibble + per-group
# 16-entry LUT); only the softmax + the standard MX requant of P are numpy. Parameterized by operand format
# (mxfp8 / mxfp6 / mxfp4).
# ------------------------------------------------------------------------------------------------
def _mx_safe_palette(mx, tok: str) -> list:
    """The requant candidate values for ``tok``: exactly-representable decoded values in the |v|<=4 window
    the operand synthesizer uses (so a requant code never overflows the 4-bit column accumulator). fp6 is
    capped to the SAME 16-value pool the synthesizer draws from (``derive_palette(...,16)``), so the fused
    PV union LUT (P-codes ∪ V-codes) stays within the 16-entry fp6 LUT."""
    from merlin.targetgen import corpus_operands as CO
    if tok == "fp6_e3m2":
        return sorted(CO.derive_palette("fp6_e3m2", 16, mag_cap=4.0))
    if tok == "fp8_e4m3":
        vals = {float(mx.fp8_e4m3_decode(c)) for c in range(256)}
    elif tok == "fp4_e2m1":
        vals = {float(mx.fp4_e2m1_decode(c)) for c in range(16)}
    else:
        raise ValueError(f"no MX palette for {tok!r}")
    return sorted(v for v in vals if v == v and abs(v) <= 4.0)


def _mx_requant_blocks(P, palette, *, group: int, target: float = 2.0):
    """Requantize float ``P[M,K]`` to (DECODED values ``[M,K]`` drawn from ``palette``, E8M0 scale codes
    ``[K/group, M]``) as the MX PE does before a GEMM: one shared power-of-two E8M0 scale per (K-group, row)
    — the (group, lane) granularity ``mx_matmul`` indexes SA with — chosen so the block max maps to
    ~``target``, then nearest-palette rounding of each scaled element. Returns DECODED values (not codes) so
    the caller re-encodes them through the SAME codec as the synth operands (fp8 byte / fp4 nibble / fp6
    LUT). The block scale is applied back by mx_matmul via the E8M0 code (2**(code-127))."""
    import math
    import numpy as np
    M, K = P.shape
    # FAIL CLOSED ON A PARTIAL BLOCK. `K // group` silently drops the elements past the last whole group,
    # and every array here is zero-initialised, so the tail comes back as zeros and the golden simply does
    # not depend on that part of its own input. MEASURED: at K=33 one column is dropped; at K=48 sixteen of
    # forty-eight are -- a THIRD of the reduction -- and perturbing A[0,32] with K=33 leaves the result
    # bit-identical. That is a silently wrong golden, which is worse than no golden: it would certify a
    # backend that also ignored the tail and fail one that did not.
    #
    # No capsule on disk trips this (every MX K is 32 or 64), so refusing here changes nothing today and
    # turns the trap into a message. It is also the reason MX coverage is aligned-only: a non-aligned MX
    # capsule cannot be minted, so the tail path has never been exercised. Supporting it means giving the
    # tail group its own E8M0 scale over a short block -- a real change to this reference, not a relaxation
    # of this guard.
    if K % group:
        raise ValueError(
            f"MX requant needs K to be a whole multiple of the {group}-element block-scale group; got "
            f"K={K} ({K % group} element(s) in a partial final group). The reference assigns one E8M0 "
            f"scale per whole group and would silently zero the tail, producing a golden that ignores "
            f"{K % group} of its own K elements. Use a K that is a multiple of {group}, or extend this "
            f"reference to scale a partial final group.")
    G = K // group
    pv = sorted(palette)
    dec = np.zeros((M, K), dtype=np.float64)
    scodes = np.zeros((G, M), dtype=np.uint8)
    for m in range(M):
        for g in range(G):
            blk = P[m, g * group:(g + 1) * group]
            mabs = float(np.max(np.abs(blk)))
            e = 0 if mabs == 0.0 else int(round(math.log2(mabs / target)))
            scodes[g, m] = max(0, min(254, e + 127))
            s = 2.0 ** (int(scodes[g, m]) - 127)
            for j in range(group):
                t = float(blk[j]) / s
                dec[m, g * group + j] = min(pv, key=lambda val: abs(val - t))
    return dec, scodes


def _mx_stage_matmul(mx, A_dec, B_dec, SA, SB, M, N, K, tok):
    """ONE MX GEMM over DECODED operands + E8M0 scales at ``tok`` (mxfp8 / mxfp6 / mxfp4), using the SAME
    codec as the R6/R7 tiles: fp8 = one code byte per element; fp4 = e2m1 nibble packed (A along rows, B
    along cols); fp6 = e3m2 nibble packed indexing a per-group union 16-entry LUT (G=log2 rows/cols per LUT
    block). Returns (C_float[M][N] bf16-decoded, packing-artifacts dict for provenance). A_dec/B_dec must be
    exactly representable in ``tok`` (synth operands + palette-requantized P both are)."""
    import numpy as np
    codes = _mx_value_codes(mx, tok)

    def enc(X):
        return np.vectorize(lambda v: codes[float(np.float32(v))])(X).astype(np.uint8)

    A_codes, B_codes = enc(A_dec), enc(B_dec)
    lutA = lutB = None
    if tok == "fp8_e4m3":
        fmt, G = mx.FMT_FP8, 0
        Ab, Bb = A_codes, B_codes
    else:
        if tok == "fp6_e3m2":
            fmt, G = mx.FMT_FP6, 5
            lut = np.array(sorted({int(c) for c in A_codes.reshape(-1)} |
                                  {int(c) for c in B_codes.reshape(-1)}), dtype=np.uint8)
            assert lut.size <= 16, f"fp6 LUT overflow ({lut.size} > 16) — requant/synth palette too wide"
            lut = np.pad(lut, (0, 16 - lut.size))[:16]
            idx = {int(v): i for i, v in enumerate(lut)}
            A_nib = np.vectorize(lambda c: idx[int(c)])(A_codes).astype(np.uint8)
            B_nib = np.vectorize(lambda c: idx[int(c)])(B_codes).astype(np.uint8)
            grp = 1 << G
            lutA = np.tile(lut.reshape(1, 16), ((A_codes.shape[0] + grp - 1) // grp, 1))
            lutB = np.tile(lut.reshape(1, 16), ((B_codes.shape[1] + grp - 1) // grp, 1))
        else:                                                    # fp4: nibble == code
            fmt, G = mx.FMT_FP4, 0
            A_nib, B_nib = A_codes, B_codes
        Ab = ((A_nib[1::2, :] << 4) | (A_nib[0::2, :] & 0xF)).astype(np.uint8)     # pack along M (rows)
        Bb = ((B_nib[:, 1::2] << 4) | (B_nib[:, 0::2] & 0xF)).astype(np.uint8)     # pack along N (cols)
    C = np.asarray(mx.mx_matmul(Ab, Bb, SA, SB, M, N, K, fmt=fmt, lutA=lutA, lutB=lutB, G=G))
    Cf = [[float(mx.bf16_to_f32(int(C[i, j]))) for j in range(N)] for i in range(M)]
    art = {"A_bytes": Ab.reshape(-1).tolist(), "A_shape": list(Ab.shape),
           "B_bytes": Bb.reshape(-1).tolist(), "B_shape": list(Bb.shape), "G": G,
           "lutA": lutA.tolist() if lutA is not None else None,
           "lutB": lutB.tolist() if lutB is not None else None}
    return Cf, art


def _mx_attention_golden(entry, binding):
    """Fused MX flash-attention golden (bf16 output) + provenance, composed from mx_ref (QK & PV, at the
    entry's operand format) + a numpy bf16 row-softmax + a per-(K-group,row) E8M0 requant of P. Shapes:
    M queries, H head dim (K of QK), Skv keys, Dv value dim. fp8 tiles by DIM=16; fp6/fp4 tile by 32, so
    for a sub-format M, Skv, Dv must all be multiples of 32 (a smaller tile yields a degenerate all-zero
    GEMM). Optional Gemma-2 logit soft-cap via ``softcap``."""
    import math

    import numpy as np

    from merlin.runtime.fp8_formats import canonical_float, e8m0_decode
    from merlin.targetgen import corpus_operands as CO
    mx = _mx_ref()
    tok = canonical_float(binding.operand_dtype)               # fp8_e4m3 / fp6_e3m2 / fp4_e2m1
    if tok not in ("fp8_e4m3", "fp6_e3m2", "fp4_e2m1"):
        raise ValueError(f"MX attention operand dtype {binding.operand_dtype!r} -> {tok!r} unsupported")
    sub = (tok != "fp8_e4m3")
    row_tile = 32 if sub else mx.DIM                           # mx_matmul TILE for this format
    max_alpha = 16 if tok == "fp6_e3m2" else None              # fp6 draws from a 16-value LUT pool
    dim = binding.tile_dim
    M = entry.get("M", entry.get("M_tiles", 1) * dim)
    H = entry.get("H", entry.get("head_dim", 2 * dim))         # QK contraction (head dim), must be %GROUP
    Skv = entry.get("Skv", entry.get("keys", 2 * dim))         # key positions
    Dv = entry.get("Dv", dim)                                  # value dim
    if H % mx.GROUP or Skv % mx.GROUP or M % row_tile or Dv % row_tile:
        raise ValueError(f"MX attention dims must satisfy H%{mx.GROUP}=Skv%{mx.GROUP}=0 and "
                         f"M%{row_tile}=Dv%{row_tile}=0 for {tok} (got M={M} H={H} Skv={Skv} Dv={Dv})")
    att_scale = float(entry.get("scale", 1.0 / math.sqrt(H)))
    softcap = entry.get("softcap")                             # Gemma-2 logit soft-cap (None to disable)
    q, k, v, out = entry.get("q", "Q"), entry.get("k", "K"), entry.get("v", "V"), entry.get("out", "Y0")

    def synth(name, shape):
        vals = CO.operand_values(shape, tok, _salt(entry["name"], name), max_alphabet=max_alpha, mag_cap=4.0)
        prob = CO.rigor_findings(vals, shape)
        if prob:
            raise AssertionError(f"non-rigorous MX attention operand {name}{shape}: {prob}")
        return np.array(vals, dtype=np.float64).reshape(shape)

    def scales(name, shape):
        sc = np.array(CO.e8m0_scale_codes(shape, _salt(entry["name"], name)), dtype=np.uint8)
        prob = CO.scale_rigor_findings(sc.tolist())
        if prob:
            raise AssertionError(f"non-rigorous E8M0 stream {name}{shape}: {prob}")
        return sc

    def bf16_round(a):
        u = np.asarray(a, dtype=np.float32).view(np.uint32)
        return ((u >> 16) << 16).view(np.float32).astype(np.float64)

    Q = synth(q, (M, H))
    K = synth(k, (Skv, H))
    V = synth(v, (Skv, Dv))
    Kt = np.ascontiguousarray(K.T)                             # device consumes K pre-transposed (K^T)

    # stage 1: S = mx_matmul(Q[M,H], K^T[H,Skv]) -> bf16 scores [M, Skv] (UNSCALED; the logit scale +
    # optional soft-cap are applied inside stage 2, in the datapath-faithful order the kernel uses).
    SA_q = scales("SA_q", (H // mx.GROUP, M))
    SB_k = scales("SB_k", (H // mx.GROUP, Skv))
    S_rows, qk_art = _mx_stage_matmul(mx, Q, Kt, SA_q, SB_k, M, Skv, H, tok)
    SB_v = scales("SB_v", (Skv // mx.GROUP, Dv))

    if not sub:
        # stages 2-5, DATAPATH-FAITHFUL (mxfp8): the EXACT flash-kernel order — a bf16 softmax over the
        # UNNORMALIZED exp-P, the online-softmax row denominator l (kernel reduction order), a per-32-block
        # e4m3 requant of the UNNORMALIZED P, the PV MX matmul, then finalize O = O_unnorm * bf16(1/bf16(l)).
        # The reference (mx_flash_ref) is validated bit-exact vs the cyclotron RTL, so the generator and the
        # kernel share ONE arithmetic — a regeneration reproduces exactly what the kernel computes.
        from merlin.targetgen import mx_flash_ref as MXF
        O_arr, _P_codes, SA_p, _l, P_dec, pv_art = MXF.flash_attention_fp8(
            mx, S_rows, V, SB_v, M=M, Skv=Skv, Dv=Dv, att_scale=att_scale, softcap=softcap)
        O_rows = [[float(O_arr[m, j]) for j in range(Dv)] for m in range(M)]
    else:
        # sub-formats (mxfp6/mxfp4): the flash kernel's e4m3 requant is not defined for these, so keep the
        # palette-requant composition unchanged (these goldens fail closed at grade time and stay identical).
        S = bf16_round(np.array(S_rows) * att_scale)
        if softcap is not None:
            cap = float(softcap)
            S = bf16_round(cap * np.tanh(S / cap))
        # bf16 row-softmax (numerically stable: subtract row max) -> P [M, Skv]
        P = np.zeros((M, Skv), dtype=np.float64)
        for m in range(M):
            r = bf16_round(S[m] - float(np.max(S[m])))
            e = bf16_round(np.exp(r))
            P[m] = bf16_round(e / float(np.sum(e)))
        # requant P into the format palette (per (K-group,row) E8M0), then O = mx_matmul(P, V)
        palette = _mx_safe_palette(mx, tok)
        P_dec, SA_p = _mx_requant_blocks(P, palette, group=mx.GROUP)  # SA_p shape [Skv/32, M]
        O_rows, pv_art = _mx_stage_matmul(mx, P_dec, V, SA_p, SB_v, M, Dv, Skv, tok)

    prov = {
        q: {"shape": [M, H], "decoded": Q.reshape(-1).tolist()},
        k: {"shape": [Skv, H], "decoded": K.reshape(-1).tolist()},
        v: {"shape": [Skv, Dv], "decoded": V.reshape(-1).tolist()},
        "SA_q_e8m0_codes": SA_q.tolist(), "SB_k_e8m0_codes": SB_k.tolist(),
        "SB_v_e8m0_codes": SB_v.tolist(),
        "scale_example": {"SA_q[0][0]": int(SA_q[0, 0]), "as_scale": e8m0_decode(int(SA_q[0, 0]))},
        # The four scale streams under the operand names the capsule declares, so a submitted backend is
        # handed them alongside the elements. P_scale is the exponent the softmax intermediate is
        # requantized against: chosen HERE when the golden was built, so the kernel cannot derive it.
        f"{q}_scale": {"shape": [H // mx.GROUP, M], "decoded": SA_q.reshape(-1).tolist(),
                       "note": "E8M0 codes, one per block of H per query row"},
        f"{k}_scale": {"shape": [H // mx.GROUP, Skv], "decoded": SB_k.reshape(-1).tolist(),
                       "note": "E8M0 codes, one per block of H per key row"},
        f"{v}_scale": {"shape": [Skv // mx.GROUP, Dv], "decoded": SB_v.reshape(-1).tolist(),
                       "note": "E8M0 codes, one per block of Skv per value column"},
        "P_scale": {"shape": [Skv // mx.GROUP, M], "decoded": SA_p.reshape(-1).tolist(),
                    "note": "E8M0 codes the softmax intermediate P is requantized against"},
        # RAW device operand bytes exactly as mx_ref consumed them (per stage, format-packed) + LUTs, so a
        # bit-exact grade re-runs the two MX GEMMs + the pinned softmax/requant over THESE codes.
        "attention_codes": {
            "q": q, "k": k, "v": v, "fmt": tok, "M": M, "H": H, "Skv": Skv, "Dv": Dv,
            "att_scale": att_scale, "softcap": (None if softcap is None else float(softcap)),
            "SA_q": SA_q.reshape(-1).tolist(), "SB_k": SB_k.reshape(-1).tolist(),
            "SB_v": SB_v.reshape(-1).tolist(), "SA_p": SA_p.reshape(-1).tolist(),
            "qk_stage": qk_art, "pv_stage": pv_art,
            # the requantized P intermediate DECODED values (derived from the softmax; NOT an input operand).
            "P_decoded": P_dec.reshape(-1).tolist(),
        },
    }
    return {out: O_rows}, prov


def _mx_gemv_batched_golden(entry, binding):
    """Batched MX matmul golden (radiance-kernels decode-time gemv_batched, MX regime): ``B`` independent
    MX GEMMs ``A_b[M,H] @ W_b[H,N]`` on the block-scaled mx_pe, stacked row-major into ``[B*M, N]`` bf16.
    (The MX PE tiles N by ``DIM``=16, so N must be a multiple of 16 — a literal N=1 gemv is not expressible
    on the mx_ref datapath; this is the faithful batched analog.) mxfp8 only; golden from mlc mx_ref."""
    import numpy as np

    from merlin.runtime.fp8_formats import canonical_float, e8m0_decode
    from merlin.targetgen import corpus_operands as CO
    mx = _mx_ref()
    tok = canonical_float(binding.operand_dtype)
    if tok != "fp8_e4m3":
        raise ValueError(f"MX gemv_batched supports mxfp8 only (got {binding.operand_dtype!r} -> {tok!r})")
    dim = binding.tile_dim
    B = int(entry.get("B", 2))
    M = entry.get("M", entry.get("M_tiles", 1) * dim)
    H = entry.get("H", entry.get("K", 2 * dim))               # contraction dim, %32
    N = entry.get("N", dim)                                    # %16
    if H % mx.GROUP or M % mx.DIM or N % mx.DIM:
        raise ValueError(f"MX gemv_batched dims: H%{mx.GROUP}=0, M%{mx.DIM}=N%{mx.DIM}=0 "
                         f"(got B={B} M={M} H={H} N={N})")
    codes = _mx_value_codes(mx, tok)
    lhs, weight, out = entry.get("lhs", "A0"), entry.get("weight", "W"), entry.get("out", "Y0")

    def synth(name, shape):
        vals = CO.operand_values(shape, tok, _salt(entry["name"], name), mag_cap=4.0)
        prob = CO.rigor_findings(vals, shape)
        if prob:
            raise AssertionError(f"non-rigorous MX gemv operand {name}{shape}: {prob}")
        return np.array(vals, dtype=np.float64).reshape(shape)

    def enc(A):
        return np.vectorize(lambda x: codes[float(np.float32(x))])(A).astype(np.uint8)

    rows_out: list = []
    A_dec, W_dec, batches = [], [], []
    for b in range(B):
        A = synth(f"{lhs}{b}", (M, H))
        W = synth(f"{weight}{b}", (H, N))
        SA = np.array(CO.e8m0_scale_codes((H // mx.GROUP, M), _salt(entry["name"], f"SA{b}")), dtype=np.uint8)
        SB = np.array(CO.e8m0_scale_codes((H // mx.GROUP, N), _salt(entry["name"], f"SB{b}")), dtype=np.uint8)
        for nm, sc in ((f"SA{b}", SA), (f"SB{b}", SB)):
            prob = CO.scale_rigor_findings(sc.tolist())
            if prob:
                raise AssertionError(f"non-rigorous E8M0 stream {nm}{sc.shape}: {prob}")
        C = np.asarray(mx.mx_matmul(enc(A), enc(W), SA, SB, M, N, H, fmt=mx.FMT_FP8))
        rows_out.extend([[float(mx.bf16_to_f32(int(C[i, j]))) for j in range(N)] for i in range(M)])
        A_dec.append(A.reshape(-1).tolist())
        W_dec.append(W.reshape(-1).tolist())
        batches.append({"A_bytes": enc(A).reshape(-1).tolist(), "W_bytes": enc(W).reshape(-1).tolist(),
                        "SA": SA.reshape(-1).tolist(), "SB": SB.reshape(-1).tolist()})
    prov = {
        lhs: {"shape": [B, M, H], "decoded": A_dec},
        weight: {"shape": [B, H, N], "decoded": W_dec},
        "batched_codes": {"lhs": lhs, "weight": weight, "fmt": tok, "B": B, "M": M, "H": H, "N": N,
                          "stacked_out_shape": [B * M, N], "batches": batches},
        "scale_example": {"SA0[0][0]": batches[0]["SA"][0],
                          "as_scale": e8m0_decode(int(batches[0]["SA"][0]))},
        # The per-batch scale streams under the operand names the capsule declares, so a submitted
        # backend is handed them the same way it is handed the elements (see the single-GEMM path).
        f"{lhs}_scale": {"shape": [B, H // mx.GROUP, M],
                         "decoded": [c for bt in batches for c in bt["SA"]],
                         "note": "E8M0 exponent codes per batch, one per block of H per lhs row"},
        f"{weight}_scale": {"shape": [B, H // mx.GROUP, N],
                            "decoded": [c for bt in batches for c in bt["SB"]],
                            "note": "E8M0 exponent codes per batch, one per block of H per weight column"},
    }
    return {out: rows_out}, prov


def _simt_golden(entry, binding):
    """SIMT (CVFPU) golden in ordinary IEEE float — fp32 accumulate, format-rounded operands. Covers the
    matmul / attention / rmsnorm shapes; independent of any accelerator model (the SIMT cores do plain IEEE
    math). Operands are format-derived + rigor-gated."""
    from merlin.runtime.fp8_formats import canonical_float
    from merlin.targetgen import corpus_operands as CO
    tok = canonical_float(binding.operand_dtype)          # fp16 / bf16 / f32
    dim = binding.tile_dim
    op = entry.get("op", "matmul")

    def q(arr):
        a = np.asarray(arr, dtype=np.float64)
        if tok == "fp16":
            return a.astype(np.float16).astype(np.float64)
        if tok == "bf16":                                 # operands are exact bf16 already; identity round
            u = a.astype(np.float32).view(np.uint32)
            return ((u >> 16) << 16).view(np.float32).astype(np.float64)
        return a.astype(np.float32).astype(np.float64)

    def synth(name, shape):
        vals = CO.operand_values(shape, tok, _salt(entry["name"], name))
        problems = CO.rigor_findings(vals, shape)
        if problems:
            raise AssertionError(f"non-rigorous SIMT operand {name}{shape}: {problems}")
        return q(np.array(vals, dtype=np.float64).reshape(shape))

    def rnd_out(y):
        return [[float(np.float32(v)) for v in row] for row in np.asarray(y)]

    prov, outputs = {}, {}
    if op in ("matmul", "linear"):
        M = entry.get("M", entry.get("M_tiles", 1) * dim)
        K = entry.get("K", entry.get("K_tiles", 1) * dim)
        N = entry.get("N", entry.get("N_tiles", 1) * dim)
        A = synth(entry.get("lhs", "A0"), (M, K))
        W = synth(entry.get("weight", "W"), (K, N))
        y = (A.astype(np.float32) @ W.astype(np.float32)).astype(np.float64)
        epi = entry.get("epilogue", [])
        if "acc_scale" in epi:
            y = y * float(entry["acc_scale"])
        if "relu" in epi:
            y = np.maximum(y, 0.0)
        prov[entry.get("lhs", "A0")] = {"shape": [M, K], "decoded": A.reshape(-1).tolist()}
        prov[entry.get("weight", "W")] = {"shape": [K, N], "decoded": W.reshape(-1).tolist()}
        outputs[entry.get("out", "Y0")] = rnd_out(y)
    elif op == "movement":
        # A load->store movement (mvin/mvout) moves data and computes nothing, so the reference is the
        # operand itself at the OUTPUT format. Its value as a capsule is that it exercises the movement
        # family the contract declares, on a datapath whose only job is to not corrupt what it carries.
        M = entry.get("M", entry.get("M_tiles", 1) * dim)
        N = entry.get("N", entry.get("N_tiles", 1) * dim)
        X = synth(entry.get("src", "X"), (M, N))
        prov[entry.get("src", "X")] = {"shape": [M, N], "decoded": X.reshape(-1).tolist()}
        outputs[entry.get("out", "Y0")] = rnd_out(X)
    elif op == "attention_qk":
        M = entry.get("M_tiles", 1) * dim
        Kd = entry.get("K_tiles", 1) * dim
        Q = synth(entry.get("q", "Q"), (M, Kd))
        Kk = synth(entry.get("k", "K"), (M, Kd))
        y = (Q.astype(np.float32) @ Kk.astype(np.float32).T).astype(np.float64)
        prov[entry.get("q", "Q")] = {"shape": [M, Kd], "decoded": Q.reshape(-1).tolist()}
        prov[entry.get("k", "K")] = {"shape": [M, Kd], "decoded": Kk.reshape(-1).tolist()}
        outputs[entry.get("out", "Y0")] = rnd_out(y)
    elif op == "rmsnorm":
        M = entry.get("M", entry.get("M_tiles", 1) * dim)
        K = entry.get("K", entry.get("K_tiles", 1) * dim)
        eps = float(entry.get("eps", 1.0 / 65536.0))
        X = synth(entry.get("src", "X"), (M, K))
        gamma = synth(entry.get("gamma", "G"), (1, K))[0]
        y = np.empty((M, K), dtype=np.float64)
        for m in range(M):
            row = X[m].astype(np.float32)
            ss = np.float32(0.0)
            for k in range(K):
                ss = np.float32(ss + np.float32(row[k] * row[k]))
            mean = np.float32(ss / np.float32(K))
            rms = np.float32(1.0) / np.float32(np.sqrt(np.float32(mean + np.float32(eps))))
            for k in range(K):
                y[m, k] = float(np.float32(np.float32(row[k] * rms) * np.float32(gamma[k])))
        prov[entry.get("src", "X")] = {"shape": [M, K], "decoded": X.reshape(-1).tolist()}
        prov[entry.get("gamma", "G")] = {"shape": [1, K], "decoded": gamma.tolist()}
        outputs[entry.get("out", "Y0")] = rnd_out(y)
    elif op == "rmsnorm_qkv":
        # fused pre-norm QKV projection: H = rmsnorm(X, gamma); Y = H @ Wqkv. Both stages IEEE fp32-accum.
        M = entry.get("M", entry.get("M_tiles", 1) * dim)
        K = entry.get("K", entry.get("K_tiles", 1) * dim)
        N = entry.get("N", entry.get("N_tiles", 1) * dim)
        eps = float(entry.get("eps", 1.0 / 65536.0))
        X = synth(entry.get("src", "X"), (M, K))
        gamma = synth(entry.get("gamma", "G"), (1, K))[0]
        Wqkv = synth(entry.get("weight", "Wqkv"), (K, N))
        Hn = np.empty((M, K), dtype=np.float64)
        for m in range(M):
            row = X[m].astype(np.float32)
            ss = np.float32(0.0)
            for k in range(K):
                ss = np.float32(ss + np.float32(row[k] * row[k]))
            rms = np.float32(1.0) / np.float32(np.sqrt(np.float32(np.float32(ss / np.float32(K)) + np.float32(eps))))
            for k in range(K):
                Hn[m, k] = float(np.float32(np.float32(row[k] * rms) * np.float32(gamma[k])))
        y = (Hn.astype(np.float32) @ Wqkv.astype(np.float32)).astype(np.float64)
        prov[entry.get("src", "X")] = {"shape": [M, K], "decoded": X.reshape(-1).tolist()}
        prov[entry.get("gamma", "G")] = {"shape": [1, K], "decoded": gamma.tolist()}
        prov[entry.get("weight", "Wqkv")] = {"shape": [K, N], "decoded": Wqkv.reshape(-1).tolist()}
        outputs[entry.get("out", "Y0")] = rnd_out(y)
    elif op == "rope_qkv":
        # fused QKV projection + RoPE: H = X @ Wqkv; Y = rope(H). GPT-NeoX/Llama rotation (theta=10000),
        # position = row index, identical convention to the pytorch RP8 rope (capsule_source._rope).
        M = entry.get("M", entry.get("M_tiles", 1) * dim)
        K = entry.get("K", entry.get("K_tiles", 1) * dim)
        N = entry.get("N", entry.get("N_tiles", 1) * dim)
        X = synth(entry.get("src", "X"), (M, K))
        Wqkv = synth(entry.get("weight", "Wqkv"), (K, N))
        H = (X.astype(np.float32) @ Wqkv.astype(np.float32)).astype(np.float64)
        half = N // 2
        theta = float(entry.get("rope_theta", 10000.0))
        freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) / half))
        pos = np.arange(M, dtype=np.float64)
        ang = pos[:, None] * freq[None, :]
        cos = np.concatenate([np.cos(ang), np.cos(ang)], axis=1)
        sin = np.concatenate([np.sin(ang), np.sin(ang)], axis=1)
        x1, x2 = H[:, :half], H[:, half:]
        rot = np.concatenate([-x2, x1], axis=1)
        y = (H.astype(np.float32) * cos.astype(np.float32)
             + rot.astype(np.float32) * sin.astype(np.float32)).astype(np.float64)
        prov[entry.get("src", "X")] = {"shape": [M, K], "decoded": X.reshape(-1).tolist()}
        prov[entry.get("weight", "Wqkv")] = {"shape": [K, N], "decoded": Wqkv.reshape(-1).tolist()}
        outputs[entry.get("out", "Y0")] = rnd_out(y)
    else:
        raise ValueError(f"no SIMT golden for op {op!r}")
    return outputs, prov


def _entry_regime(entry, binding):
    """Route an entry to its numeric regime + return a per-entry binding (operand/accum overridden). ``int``
    (gemmini), ``specir`` (atlas fp8), ``mx`` (microscaling block-scaled FP), ``simt`` (IEEE fp16/bf16/f32).
    Routed purely by the entry's operand dtype token — no target name."""
    from merlin.runtime.fp8_formats import canonical_float
    tok = entry.get("operand_dtype") or binding.operand_dtype
    if CS.is_block_scaled(tok):
        regime, acc = "mx", "bf16"
    else:
        try:
            canon = canonical_float(tok)
        except KeyError:
            canon = None
        if canon in ("fp8_e4m3", "fp8_e5m2"):
            regime, acc = "specir", binding.accum_dtype
        elif canon in ("fp16", "bf16", "f32"):
            regime, acc = "simt", "f32"
        else:
            regime, acc = "int", binding.accum_dtype
    eb = dataclasses.replace(
        binding, operand_dtype=tok, accum_dtype=acc, integer=(regime == "int"),
        compare=("exact_int" if regime == "int" else "tolerance_float"))
    return regime, eb


# ------------------------------------------------------------------------------------------------
def _write_capsule(entry, binding, out_root):
    """Write one capsule, then GUARANTEE it carries its generalization-intent block.

    The stamp is a post-step rather than something each writer does, because there are four writers
    (direct-MLIR, pytorch-sourced, spec-sourced, whole-model) and three of them build their capsule dict
    themselves and return early. Stamping inside ``corpus_spec.build`` alone left 14 of atlas's 33
    capsules unannotated -- exactly the silent-gap failure mode this block exists to close -- so it is
    applied here, at the one point every path must pass through.
    """
    written = _write_capsule_inner(entry, binding, out_root)
    if not written:
        return written
    d = Path(written) if not isinstance(written, Path) else written
    capf = d / "capsule.yaml" if d.is_dir() else None
    if capf is None or not capf.exists():
        return written
    cap = yaml.safe_load(capf.read_text()) or {}
    dirty = False
    if not (cap.get("semantic") or {}).get("generalization_axis"):
        _, eb = _entry_regime(entry, binding)
        cap["semantic"] = CS._semantic_block(entry, eb)
        dirty = True
    dirty = _backfill_required_classes(cap, binding) or dirty
    _validate_lane_declaration(entry, binding)
    dirty = _carry_declared_blocks(entry, cap) or dirty
    # THE TOLERANCE MUST BE FALSIFIABLE AT THIS GOLDEN'S SCALE, and here is the first point at which
    # both the capsule and its golden exist for EVERY writer -- the same reason the generalization stamp
    # lives here. A profile declares ONE absolute tolerance for a whole target, which is the right shape
    # for a datapath error budget and the wrong shape for a small-magnitude output: a softmax capsule
    # whose golden spans 0.0139..0.1523 was graded at `atol: 0.25`, so zeros, the mean and the midrange
    # all passed it. It reported a numeric pass and proved nothing.
    _gp = d / "golden.yaml"
    if _gp.is_file() and (cap.get("numeric_policy") or {}).get("atol") is not None:
        _gdoc = yaml.safe_load(_gp.read_text(encoding="utf-8")) or {}
        _pol, _prov = NF.falsifiable_policy(cap["numeric_policy"], _gdoc.get("outputs") or {},
                                            name=str(entry.get("name") or d.name))
        if cap.get("numeric_policy") != _pol or cap.get("numeric_falsifiability") != _prov:
            cap["numeric_policy"] = _pol
            cap["numeric_falsifiability"] = _prov
            dirty = True
    if dirty:
        capf.write_text(yaml.safe_dump(cap, sort_keys=False), encoding="utf-8")
    _write_capsule_readme(entry, cap, d)
    return written


#: Profile-entry keys that describe what a capsule is FOR rather than what it computes, and which every
#: writer must carry through untouched. They are stamped in the same post-step as the generalization
#: block, and for the same reason: three of the four writers build their capsule dict themselves, so a
#: key handled in only one of them is silently absent from two thirds of the corpus.
#:
#: ``performance``      which optimization level the capsule exercises and which schedule lever its cycle
#:                      count can see. A capsule is otherwise mute about this, so a perf corpus and a
#:                      functional corpus are indistinguishable once generated.
#: ``comparison_group`` the capsule's place in a set whose cycle counts are comparable to one another --
#:                      a fused implementation against the parts it replaces. The field has been declared
#:                      on four capsules since they were written and consumed by nothing, which is the
#:                      same thing as not existing.
#: ``pass_requirements`` the compiler-obligation classes a capsule demands, which is the ONLY link
#:                      between a catalogued pass and a concrete capsule that requires it
#:                      (``check_pass_obligations.py`` rejects a pass no capsule obliges). It was
#:                      hand-written onto two capsules and unknown to this generator, so every
#:                      regeneration silently deleted the corpus's only pass obligations.
#: ``lanes``               the interop/negative-lane contract: which execution lanes must have carried
#:                      work, and (``forbid``) which must have carried none. Only the whole-model writer
#:                      emitted it, so a model_slice capsule declaring lanes silently lost them -- which
#:                      is how the first host-only capsule generated with `lanes: None` and asserted
#:                      nothing at all.
_DECLARED_BLOCKS = ("performance", "comparison_group", "pass_requirements", "lanes")


def _carry_declared_blocks(entry: dict, cap: dict) -> bool:
    """Copy the profile entry's declared intent blocks onto the capsule. Never overwrites one already
    there (a hand-authored capsule is the source of record), and never invents one."""
    dirty = False
    for key in _DECLARED_BLOCKS:
        value = entry.get(key)
        if value is None or cap.get(key) is not None:
            continue
        cap[key] = dict(value) if isinstance(value, dict) else value
        dirty = True
    return dirty


def _validate_lane_declaration(entry: dict, binding) -> None:
    """Refuse an unreachable or self-contradictory lane declaration AT GENERATION TIME.

    The whole-model writer already ran ``_checked_lanes``; the other writers did not, because they never
    carried lanes at all. Now that every writer does, the check has to move with it -- a bar the target's
    declared units make unreachable is not a capability test, it is a wall, and the place to catch it is
    where an author can still fix it.
    """
    lanes = entry.get("lanes") or {}
    if not lanes:
        return
    from merlin.targetgen.capsule_source import _checked_lanes
    _checked_lanes(entry, binding)                      # raises on an unreachable `require`
    forbid = [str(x) for x in (lanes.get("forbid") or ())]
    both = sorted(set(str(x) for x in (lanes.get("require") or ())) & set(forbid))
    if both:
        raise ValueError(f"{entry.get('name')!r}: lane(s) {both} are both required and forbidden; one "
                         f"of the two assertions can never hold")
    target = getattr(binding, "target", None)
    if forbid and target:
        from merlin.targetgen.routing import reachable_lanes
        unreachable = sorted(set(forbid) - reachable_lanes(target))
        if unreachable:
            raise ValueError(
                f"{entry.get('name')!r}: forbids lane(s) {unreachable} that {target!r} cannot populate "
                f"anyway, so the assertion is vacuously true and tests nothing")


def _write_capsule_readme(entry: dict, cap: dict, d: Path) -> None:
    """Write the capsule's ``README.md`` -- the 5th of the five files a capsule is DEFINED to have.

    The generator only ever emitted four of them, so every generated capsule was incomplete by the
    corpus's own definition and the materialized public view failed its own completeness check the moment
    a capsule arrived without a hand-written README. Derived from the profile entry and the capsule, so it
    cannot go stale: the prose is the entry's ``comment`` when it has one, otherwise a sentence built from
    the op, the source it was authored from, and the operand shapes/dtypes. Never overwrites a README that
    is already there -- the hand-written ones are the frozen source-of-record."""
    rd = d / "README.md"
    if rd.exists():
        return
    name = cap.get("name") or entry.get("name", "")
    prose = (entry.get("comment") or "").strip()
    if not prose:
        op = (cap.get("operation") or {}).get("op") or entry.get("op") or "unknown"
        ops = ", ".join(f"{i.get('name')}{list(i.get('shape') or [])}:{i.get('dtype')}"
                        for i in (cap.get("inputs") or []) if i.get("name"))
        src = cap.get("source_reference") or entry.get("source_reference") or ""
        prose = f"{name}: {op}" + (f" over {ops}" if ops else "")
        prose += f", authored from {src}." if src else "."
    line = " ".join(f"{k}={v}" for k, v in (
        ("kind", cap.get("kind") or entry.get("kind")),
        ("label", cap.get("label") or entry.get("label")),
        ("op", (cap.get("operation") or {}).get("op") or entry.get("op")),
        ("modes", (cap.get("expected") or {}).get("modes", {})),
    ) if v is not None)
    rd.write_text(f"# {name}\n\n{prose}\n\n{line}\n", encoding="utf-8")


def _backfill_required_classes(cap: dict, binding) -> bool:
    """Fill an EMPTY ``expected.instruction_classes`` from the target's own derived taxonomy.

    The source-backed writers (pytorch / spec / model) build their capsule dict themselves and leave this
    empty, so a contraction authored in PyTorch shipped with no coverage requirement at all while the
    direct-MLIR twin next to it carried the full systolic sequence -- the L1 coverage assertion silently
    did not apply to exactly the frontend-faithful capsules the generalization corpus is made of.

    Derived, never hardcoded: the slots come from the op's family in the closed vocabulary and are mapped
    to class names through THIS target's role census. Fail-closed at every step -- an op that owes no
    contraction, an undecidable taxonomy, or a role the target does not have all leave the list empty
    rather than inventing a demand. Only ever fills an empty list; never edits an authored one."""
    exp = cap.get("expected")
    if not isinstance(exp, dict) or exp.get("instruction_classes"):
        return False
    op = (cap.get("operation") or {}).get("op")
    if not op:
        return False
    attrs = (cap.get("operation") or {}).get("attributes", {}) or {}
    modes = exp.get("modes", {}) or {}
    from merlin.targetgen import isa_taxonomy as IT
    tax = IT.taxonomy_for_target(binding.target)   # {} when the target ships no ISA definition
    if not tax or not tax.get("by_class"):
        return False
    want = IT.required_classes_for_op(
        tax, op=op,
        output_dtype=attrs.get("output_dtype") or (cap.get("numeric_policy") or {}).get("dtype"),
        epilogue=tuple(attrs.get("epilogue", []) or []),
        movement=op in ("movement", "copy") or bool(modes.get("movement")),
    )
    if not want:
        return False
    exp["instruction_classes"] = list(want)
    return True


def _write_capsule_inner(entry, binding, out_root):
    regime, eb = _entry_regime(entry, binding)
    # Whole-model capsule: a small representative network lowered end-to-end via model2MLIR, graded vs its
    # host torch-eager output, GATED so it runs only after the op suite proves itself. Additive: skipped
    # (loudly) when the m2m venv is absent.
    if entry.get("kind") == "model" or entry.get("op") == "model":
        from merlin.targetgen import capsule_source as CSRC
        src = CSRC.PytorchRefSource()
        if not src.available():
            print(f"  [skip] {entry['name']}: model capsule needs the m2m venv (set MERLIN_M2M_PYTHON)")
            return None
        return CSRC.write_model_capsule(entry, eb, out_root, source=src)
    # PREFERRED source: a capsule defined in PyTorch (frontend-faithful), lowered to linalg via model2MLIR
    # with a host torch-eager golden. Opt in per entry (``source: pytorch``). Restricted to the float
    # regime: a host-eager float reference is graded with tolerance, matching the merlin_iface float
    # interface; int/MX datapaths keep the direct-MLIR engines below (the endorsed fallback for the
    # dtypes torch/torchAO does not faithfully model, e.g. int8xint8 systolic or block-scaled MX).
    if entry.get("source") == "pytorch" or entry.get("pytorch_ref"):
        # AN ENTRY THAT NAMES A QUANTIZATION SCHEME has said which arithmetic its program must contain,
        # so the float-regime restriction below does not apply to it. The restriction exists because a
        # host-eager float reference cannot grade an int/MX datapath -- but that is a statement about
        # the DEFAULT weight-only capture, which emits a float matmul over dequantized weights. A W8A8
        # scheme emits `aten._int_mm` accumulating in i32, which IS the mesh's arithmetic, and torch
        # eager then computes the same quantized math, so the golden is right by construction.
        if entry.get("quant_scheme"):
            pass
        elif regime != "simt":
            raise ValueError(f"pytorch source for capsule {entry['name']!r} needs a float dtype "
                             f"(got regime {regime!r} for {eb.operand_dtype!r}); author int/MX capsules "
                             f"via the direct-MLIR engine")
        from merlin.targetgen import capsule_source as CSRC
        src = CSRC.PytorchRefSource()
        if not src.available():
            # A pytorch capsule needs the m2m venv (torch) at generation time. It is additive: skip it
            # (loudly) rather than sink the whole target, so a checkout without the venv still regenerates
            # the direct-MLIR corpus. A capture that STARTS but fails (opaque/crash) still raises.
            print(f"  [skip] {entry['name']}: pytorch source needs the m2m venv (set MERLIN_M2M_PYTHON)")
            return None
        return CSRC.write_pytorch_capsule(entry, eb, out_root, source=src)
    # Spec source: a capsule whose PROGRAM + bit-exact golden come from the specir verification spec itself
    # (``spec_ref: '<gen>:op.<name>'``). Additive: a gen without a specir program emitter (or no specir) is
    # skipped loudly rather than sinking the target.
    if entry.get("source") == "spec" or entry.get("spec_ref"):
        from merlin.targetgen import capsule_source as CSRC
        src = CSRC.SpecRefSource()
        if not src.available():
            print(f"  [skip] {entry['name']}: spec source needs specir (set SPECIR_ROOT)")
            return None
        try:
            return CSRC.write_spec_capsule(entry, eb, out_root, source=src)
        except CSRC.SpecProgramUnavailable as e:
            print(f"  [skip] {entry['name']}: {e}")
            return None
    cap, mlir = CS.build(entry, eb)
    d = Path(out_root) / entry["cat"] / entry["name"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "capsule.yaml").write_text(yaml.safe_dump(cap, sort_keys=False), encoding="utf-8")
    (d / "capsule.interface.mlir").write_text(mlir, encoding="utf-8")
    (d / "expected_instruction_coverage.yaml").write_text(
        yaml.safe_dump(cap["expected"], sort_keys=False), encoding="utf-8")
    if regime == "int":
        (d / "golden.yaml").write_text(yaml.safe_dump(
            {"golden_source": "merlin_tensor_int", "outputs": CG.golden({**cap, "__dir__": ""})},
            sort_keys=False), encoding="utf-8")
    elif regime == "specir":
        outputs, prov = _float_golden(entry, eb)
        (d / "golden.yaml").write_text(yaml.safe_dump({
            "golden_source": "specir_refmodel_fp8_bf16",
            "oracle_provenance": {
                "engine": "specir.oracle.dtypes + specir.oracle.refmodel.fp_reduce",
                "datapath": "acc <- round_bf16(acc + round_bf16(a*w)); k index_sequential; per_step; rne",
                # How the datapath decodes an operand code. A unit that admits only normal operands
                # reads a zero exponent field as zero; the golden decodes it the same way, so the two
                # references implement ONE datapath (see the target's profile ``datapath`` block).
                "operand_decode": ("subnormal_flush_to_zero" if eb.subnormal_operand_flush
                                   else "exact"),
                "operand_dtype": eb.cap_dtype(eb.operand_dtype),
                "accum_dtype": eb.cap_dtype(eb.accum_dtype), "output_dtype": "bf16",
                "note": "INDEPENDENT of the target RTL (not self-oracle); specir refmodel is the reference.",
                "grade_policy": {"compare": eb.compare, "atol": eb.atol, "rtol": eb.rtol},
                "inputs": prov},
            "outputs": outputs}, sort_keys=False), encoding="utf-8")
    elif regime == "mx":
        # matmul/linear -> the single MX GEMM golden; attention_mx -> the fused flash-attention composition
        # (two MX GEMMs + a bf16 softmax), both over the SAME validated mx_ref engine.
        if entry.get("op") == "attention_mx":
            outputs, prov = _mx_attention_golden(entry, eb)
            engine = ("mlc.validate.mx_ref.mx_matmul x2 (QK & PV, transcribed from radiance-kernels "  # target-ok: provenance string (source repo radiance-kernels), not control flow
                      "lib/golden/mx_golden.cpp) + numpy bf16 row-softmax; P requantized to mxfp8 per-row")
            datapath = ("O = mx_matmul(softmax(mx_matmul(Q,K^T)/sqrt(H) [+softcap]), V); E8M0 per 32-elt "
                        "K group; bf16 accumulate + bf16 softmax")
        elif entry.get("op") == "gemv_batched":
            outputs, prov = _mx_gemv_batched_golden(entry, eb)
            engine = ("mlc.validate.mx_ref.mx_matmul x B (independent batched MX GEMMs stacked row-major)")
            datapath = ("B x [M,H]@[H,N] on the mx_pe; one E8M0 scale per 32-elt K group; bf16 accumulate")
        else:
            outputs, prov = _mx_golden(entry, eb)
            engine = ("mlc.validate.mx_ref.mx_matmul (transcribed from radiance-kernels "  # target-ok: provenance string (source repo radiance-kernels), not control flow
                      "lib/golden/{mx_fp_math.h,mx_golden.cpp}; mirrors the RTL, bit-exact vs spike)")
            datapath = ("16-deep systolic per-column acc schedule (ACC_E/ACC_M); one E8M0 scale per "
                        "32-elt K group; bf16 accumulate")
        (d / "golden.yaml").write_text(yaml.safe_dump({
            "golden_source": "mlc_mx_ref_hardware_semantics",
            "oracle_provenance": {
                "engine": engine,
                "datapath": datapath,
                "operand_dtype": eb.cap_dtype(eb.operand_dtype), "block_scale": "e8m0", "output_dtype": "bf16",
                "note": "NOT specir (specir is atlas fp8); MX is a distinct block-scaled datapath.",  # target-ok: descriptive note contrasting atlas-fp8 vs mx datapath, not control flow
                "grade_policy": {"compare": eb.compare, "atol": eb.atol, "rtol": eb.rtol},
                "inputs": prov},
            "outputs": outputs}, sort_keys=False), encoding="utf-8")
    else:                                                     # simt (IEEE fp16/bf16/f32)
        outputs, prov = _simt_golden(entry, eb)
        (d / "golden.yaml").write_text(yaml.safe_dump({
            "golden_source": "ieee_simt_f32_accumulate",
            "oracle_provenance": {
                "engine": "numpy IEEE float (CVFPU fp32 accumulate; format-rounded operands)",
                "operand_dtype": eb.cap_dtype(eb.operand_dtype), "accum_dtype": "f32", "output_dtype": "f32",
                "note": "SIMT cores do ordinary IEEE math; reference is independent of any accelerator model.",
                "grade_policy": {"compare": eb.compare, "atol": eb.atol, "rtol": eb.rtol},
                "inputs": prov},
            "outputs": outputs}, sort_keys=False), encoding="utf-8")
    return d


#: Extent tokens a sweep axis may use, resolved against the binding's tile edge.
#: Spelled the way `kernels/opu_corpus.py` already spells tile-relative extents
#: ("tile", "tile/2"), so one convention covers both corpora.
_TILE_TOKEN = "tile"

#: A sweep may not silently become a thousand capsules. Exceeding this raises;
#: it is never truncated, because a corpus that quietly dropped points reads as
#: "covered everything" when it did not.
_MAX_SWEEP_CAPSULES = 128


#: Entry keys whose value is a SHAPE EXTENT, and may therefore be written tile-relative.
_EXTENT_KEYS = ("M", "K", "N", "H", "Skv", "Dv", "B")


def _resolve_flat_extents(entry: dict, binding) -> dict:
    """Resolve tile-relative extent tokens on a FLAT capsule entry.

    ``resolve_extent`` was reachable only from sweep axes, so a flat entry had to spell its shape as
    integers -- which bakes one target's geometry into a file that is supposed to describe a shape
    RELATIVE to whatever edge the hardware has. Synthesized entries are written ``tile`` / ``tile-1``
    precisely so the same entry means the same thing on a target with a different edge, and this is
    where that promise is kept.

    Inert on every existing entry: an int passes through ``resolve_extent`` unchanged, and a key that
    is absent is not touched.
    """
    tile = getattr(binding, "tile_dim", None)
    if not tile:
        return entry
    out = None
    for key in _EXTENT_KEYS:
        value = entry.get(key)
        if not isinstance(value, str):
            continue
        if out is None:
            out = dict(entry)
        out[key] = resolve_extent(value, int(tile))
    return out if out is not None else entry


def resolve_extent(token, tile: int) -> int:
    """Resolve a sweep extent token against *tile* (the binding's tile edge).

    Accepts a plain int, or a tile-relative expression of the form
    ``[<mult>*]tile[+<n>|-<n>|/<div>]`` — e.g. ``tile``, ``tile-1``, ``tile+1``,
    ``tile/2``, ``2*tile``, ``2*tile-1``. Parsed structurally (``partition``), not
    by pattern-matching, so an unsupported spelling raises instead of silently
    resolving to something plausible.

    The point of tile-relative tokens is that a profile never hardcodes a
    geometry: the same sweep produces edge cases for a 16-wide command-buffer
    tile and for a 64-wide VLMAX tile without being edited.
    """
    if isinstance(token, bool):
        raise ValueError(f"sweep extent {token!r} is a bool, not an extent")
    if isinstance(token, int):
        if token < 1:
            raise ValueError(f"sweep extent {token!r} must be >= 1")
        return token
    if not isinstance(token, str):
        raise ValueError(f"sweep extent {token!r} is neither an int nor a tile expression")

    text = token.strip()
    mult_text, star, rest = text.partition("*")
    if star:
        mult = int(mult_text.strip())
        rest = rest.strip()
    else:
        mult, rest = 1, text

    for op in ("+", "-", "/"):
        head, found, tail = rest.partition(op)
        if found:
            if head.strip() != _TILE_TOKEN:
                raise ValueError(f"sweep extent {token!r}: expected {_TILE_TOKEN!r} before {op!r}")
            operand = int(tail.strip())
            base = mult * tile
            if op == "+":
                value = base + operand
            elif op == "-":
                value = base - operand
            else:
                if operand == 0:
                    raise ValueError(f"sweep extent {token!r}: division by zero")
                value = base // operand
            break
    else:
        if rest.strip() != _TILE_TOKEN:
            raise ValueError(f"sweep extent {token!r} is not a recognized tile expression")
        value = mult * tile

    if value < 1:
        raise ValueError(f"sweep extent {token!r} resolves to {value} at tile={tile}; must be >= 1")
    return value


def _performance_facts(target: str) -> dict:
    """Canonical tri-state performance facts, derived once for this target."""
    document = derive_profile(target).to_dict()
    return {
        "target": target,
        "traits": document["traits"],
        "sha256": _document_digest(document),
    }


def evaluate_gate(gate: dict, trait_facts: "dict | None") -> tuple[bool, dict]:
    """Require every canonical trait to be exactly ``True`` and retain evidence.

    The structured decision deliberately carries False and None separately. A
    refuted capability makes a family inapplicable; an unestablished one means
    the instrument/fact coverage is incomplete. Neither is admitted, and
    neither is silently reduced to Python truthiness.
    """
    if not isinstance(gate, dict):
        raise ValueError("performance gate must be a mapping")
    if "requires" in gate:
        raise ValueError("performance gate.requires is not accepted; use canonical gate.traits")
    names = gate.get("traits")
    if (not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not name for name in names)):
        raise ValueError("performance gate.traits must be a non-empty list")
    unknown = sorted(set(names) - set(TRAITS))
    if unknown:
        raise ValueError(f"unknown performance trait(s) {unknown}; canonical traits are {list(TRAITS)}")
    facts = (trait_facts or {}).get("traits", trait_facts or {})
    selected: dict[str, dict] = {}
    for name in names:
        raw = facts.get(name)
        if not isinstance(raw, dict):
            raw = {
                "satisfied": None,
                "tier": "not_established",
                "evidence": "canonical trait fact was not supplied",
                "missing": ["derive_profile(target) result for this trait"],
            }
        selected[name] = {
            "satisfied": raw.get("satisfied") if raw.get("satisfied") in (True, False) else None,
            "tier": raw.get("tier") or "not_established",
            "evidence": raw.get("evidence") or "no evidence recorded",
            "missing": list(raw.get("missing") or []),
        }
    refuted = [name for name, fact in selected.items() if fact["satisfied"] is False]
    unestablished = [name for name, fact in selected.items() if fact["satisfied"] is None]
    satisfied = [name for name, fact in selected.items() if fact["satisfied"] is True]
    outcome = "refuted" if refuted else ("unestablished" if unestablished else "satisfied")
    decision = {
        "outcome": outcome,
        "required_traits": list(names),
        "satisfied": satisfied,
        "refuted": refuted,
        "unestablished": unestablished,
        "facts": selected,
    }
    return outcome == "satisfied", decision


def _materialize_performance_entry(entry: dict, binding) -> dict:
    """Resolve a performance member onto a runnable direct corpus path.

    Dtypes come from workload_gen's capability-manifest accessor and must agree
    with the corpus binding selected for the same target.  The shared template
    therefore contains neither a target dtype nor a frontend choice.
    """
    target = str(getattr(binding, "target", "") or "")
    if not target:
        raise ValueError("performance materialization needs binding.target")
    binding_operand = getattr(binding, "operand_dtype", None)
    binding_accum = getattr(binding, "accum_dtype", None)
    if binding_operand and binding_accum:
        # This is the binding _write_capsule will actually consume, derived by
        # corpus_spec from the target experiment + numeric profile. Prefer it
        # over re-deriving through a second capability-manifest representation.
        operand_dtype, accum_dtype = str(binding_operand), str(binding_accum)
        datatype_basis = "corpus_binding"
    else:
        operand_dtype, accum_dtype = WG.datapath_formats(target, accum_dtype=binding_accum)
        datatype_basis = "merlin.perf.workload_gen.datapath_formats"
    op = str(entry.get("op") or "")
    if op not in CS.BUILDERS:
        raise ValueError(
            f"performance member {entry.get('name', '?')}: direct corpus source has no runnable "
            f"builder for {op!r}")
    entry["source"] = "direct"
    entry["operand_dtype"] = operand_dtype
    performance = copy.deepcopy(entry["performance"])
    performance["emitter"] = copy.deepcopy(performance["emitter"])
    performance["emitter"]["resolved"] = {
        "source": "direct",
        "operand_dtype": operand_dtype,
        "accum_dtype": accum_dtype,
        "datatype_basis": datatype_basis,
        "builder": "merlin.targetgen.corpus_spec.build",
    }
    entry["performance"] = performance
    return entry


def expand_sweeps(profile: dict, binding, *, trait_facts: "dict | None" = None,
                  skipped: "list | None" = None,
                  blocked_unimplemented: "list | None" = None,
                  errors: "list | None" = None,
                  traits: "dict | None" = None) -> list[dict]:
    """Return the profile's capsule entries with any ``sweeps:`` block expanded.

    A performance family's ``performance.gate.traits`` are evaluated against
    :func:`derive_profile`'s canonical tri-state records. Every trait must be
    exactly True. False and None both skip admission but remain distinct, with
    their evidence and evidence tier, in ``skipped``.

    A sweep is a cross-product over named axes plus a shared ``base``, producing
    exactly the flat entry dicts the per-capsule pipeline already consumes — so
    ``_write_capsule`` and every golden path are untouched by this feature.

    Two rules are enforced rather than documented:

    * **Every fitted axis needs at least two distinct points.** K is always
      fitted when present because one reduction depth cannot separate a tiled
      unit's rate from fixed overhead. Other axes opt in through ``fit_axes``;
      a one-point fit prices a parameter confidently and wrongly.
    * **Names must be unique across generated and hand-authored entries.** A
      collision would have one capsule overwrite another's directory, silently
      shrinking the corpus.

    Hand-written entries in ``capsules:`` are kept verbatim and come first, so a
    profile can mix a sweep with cases whose prose is worth writing by hand.
    """
    entries = list(profile.get("capsules") or [])
    sweeps = profile.get("sweeps") or []
    if not sweeps:
        return entries
    # Compatibility for the public/holdout disjointness checker, which passes
    # this old keyword even for purely functional profiles. It is never used to
    # admit performance: the first performance sweep below rejects it.
    legacy_traits_supplied = traits is not None

    tile = int(getattr(binding, "tile_dim", 0) or 0)
    if tile < 1:
        raise ValueError("sweeps need a tile edge; the binding reports none")

    seen = {e.get("name") for e in entries if isinstance(e, dict)}
    generated: list[dict] = []

    for sweep in sweeps:
        if not isinstance(sweep, dict):
            raise ValueError(f"sweep entry {sweep!r} is not a mapping")
        sweep_id = str(sweep.get("id") or "").strip()
        if not sweep_id:
            raise ValueError("every sweep needs an `id` (it prefixes the generated names)")
        base = dict(sweep.get("base") or {})
        variant_performance = [i for i, variant in enumerate(sweep.get("variants") or [])
                               if isinstance(variant, dict) and "performance" in variant]
        if variant_performance:
            raise ValueError(
                f"sweep {sweep_id!r}: performance blocks belong on `base`, not variants "
                f"{variant_performance}; every member of a family shares one claim contract")
        performance = base.get("performance")
        is_performance = base.get("cat") in {"perf", "_perf"} or performance is not None
        if is_performance:
            if legacy_traits_supplied:
                raise ValueError(
                    f"performance sweep {sweep_id}: legacy ad-hoc `traits` cannot gate performance; "
                    "pass canonical derive_profile(target) records through `trait_facts`")
            performance = _validate_performance_block(
                performance, owner=f"performance sweep {sweep_id}")
            if performance["family"] != sweep_id:
                raise ValueError(
                    f"performance sweep {sweep_id}: performance.family must equal the sweep id")
            facts = trait_facts
            if facts is None:
                target = str(getattr(binding, "target", "") or "")
                if not target:
                    raise ValueError(
                        f"performance sweep {sweep_id}: no trait facts supplied and binding.target is absent")
                facts = _performance_facts(target)
            ok, decision = evaluate_gate(performance["gate"], facts)
            if not ok:
                if skipped is not None:
                    skipped.append({
                        "family": sweep_id,
                        "sweep": sweep_id,
                        "status": "skipped_inapplicable",
                        "gate": decision,
                        "fit_axes": list(sweep.get("fit_axes") or []),
                        "comparison_roles": _comparison_roles(sweep),
                    })
                continue
            emitter_status = str(performance["emitter"]["status"])
            if emitter_status != "existing":
                if blocked_unimplemented is not None:
                    blocked_unimplemented.append({
                        "family": sweep_id,
                        "status": "blocked_unimplemented",
                        "reason": f"declared emitter {emitter_status!r} is not implemented",
                        "emitter": copy.deepcopy(performance["emitter"]),
                        "fit_axes": list(sweep.get("fit_axes") or []),
                        "comparison_roles": _comparison_roles(sweep),
                    })
                continue
        axes = sweep.get("axes") or {}
        if not isinstance(axes, dict) or not axes:
            raise ValueError(f"sweep {sweep_id!r} declares no axes")

        # Resolve each axis to concrete extents, preserving declaration order so
        # the generated corpus is reproducible.
        resolved: dict[str, list[int]] = {}
        for axis, tokens in axes.items():
            if not isinstance(tokens, list) or not tokens:
                raise ValueError(f"sweep {sweep_id!r} axis {axis!r} must be a non-empty list")
            resolved[axis] = [resolve_extent(t, tile) for t in tokens]

        declared_fit_axes = sweep.get("fit_axes") or []
        if not isinstance(declared_fit_axes, list) or any(
                not isinstance(axis, str) or not axis for axis in declared_fit_axes):
            raise ValueError(f"sweep {sweep_id!r}: `fit_axes` must be a list of axis names")
        fitted_axes = list(dict.fromkeys(
            (["K"] if "K" in resolved else []) + declared_fit_axes))
        unknown_fit_axes = [axis for axis in fitted_axes if axis not in resolved]
        if unknown_fit_axes:
            raise ValueError(
                f"sweep {sweep_id!r} fits undeclared axis/axes {unknown_fit_axes}; "
                f"declared axes are {sorted(resolved)}")
        for axis in fitted_axes:
            if len(set(resolved[axis])) < 2:
                detail = ("a tiled unit needs at least TWO distinct K points, because one cannot "
                          "separate the rate from the per-tile overhead" if axis == "K" else
                          "every fitted parameter needs at least TWO distinct points")
                raise ValueError(
                    f"sweep {sweep_id!r} fits {axis} over {resolved[axis]} — {detail}")

        combos = _cross_product(resolved)
        if len(combos) > _MAX_SWEEP_CAPSULES:
            raise ValueError(
                f"sweep {sweep_id!r} would generate {len(combos)} capsules (cap {_MAX_SWEEP_CAPSULES}); "
                f"narrow the axes rather than letting it be truncated")

        # A sweep may also cross the extents with a list of NON-EXTENT overrides. The motivating case is
        # a fusion comparison: three capsules that must sit at the IDENTICAL shape and differ only in
        # which op they ask for, so that `cycles(fused)` and `cycles(part) + cycles(part)` are about the
        # same work. Expressing that as an axis is impossible -- an axis value is an extent, resolved
        # against the tile -- and expressing it as three hand-authored entries would hardcode the shape
        # in three places, where the whole point is that the three shapes are the same one.
        variants = sweep.get("variants") or [{}]
        if not isinstance(variants, list) or any(not isinstance(v, dict) for v in variants):
            raise ValueError(f"sweep {sweep_id!r}: `variants` must be a list of mappings")

        # A comparison group whose members cannot be compared is a declaration with no content. The
        # field was carried on four shipped capsules for a year while every one of them sat alone in
        # its group, so nothing could ever do the arithmetic it exists for.
        _groups: dict[str, int] = {}
        for v in variants:
            g = v.get("comparison_group")
            gname = g.get("name") if isinstance(g, dict) else g
            if gname:
                _groups[str(gname)] = _groups.get(str(gname), 0) + 1
        _lonely = sorted(g for g, n in _groups.items() if n < 2)
        if _lonely:
            raise ValueError(
                f"sweep {sweep_id!r} declares comparison group(s) {_lonely} with a single member; a "
                "group of one cannot be compared to anything, so declare the other members or drop "
                "the group")

        template = str(sweep.get("name") or "{id}_{i:02d}")
        index = 0
        for combo in combos:
            for variant in variants:
                entry = copy.deepcopy(base)
                entry.update(combo)
                entry.update(_render_variant(variant, combo))
                entry["name"] = template.format(id=sweep_id, i=index, **{**combo, **variant})
                index += 1
                entry.setdefault("source_role", "derived_sweep")
                # Provenance survives generation: say which sweep and which point.
                reference = sweep.get("source_reference") or f"generated by sweep {sweep_id!r}"
                axis_note = ", ".join(f"{k}={v}" for k, v in sorted(combo.items()))
                if variant:
                    axis_note += "; " + ", ".join(f"{k}={v}" for k, v in sorted(variant.items())
                                                  if not isinstance(v, dict))
                entry["source_reference"] = f"{reference} (tile={tile}; {axis_note})"
                if entry["name"] in seen:
                    raise ValueError(
                        f"sweep {sweep_id!r} generated duplicate capsule name {entry['name']!r}")
                seen.add(entry["name"])
                if is_performance:
                    try:
                        _materialize_performance_entry(entry, binding)
                    except Exception as exc:  # noqa: BLE001 - persisted as a generation error
                        if errors is None:
                            raise
                        errors.append({
                            "family": sweep_id,
                            "member": entry["name"],
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "detail": str(exc)[:500],
                        })
                        continue
                generated.append(entry)

    return entries + generated


def _render_variant(variant: dict, combo: dict) -> dict:
    """A variant override with its string fields resolved against the shape point it is paired with.

    Only ``{extent}`` substitution, one level into a nested mapping -- enough for a comparison group to
    name the shape its members share (``fmb_{M}x{K}x{N}``) without any entry writing that shape down.
    A field with no placeholder is copied through untouched.
    """
    out: dict = {}
    for key, value in variant.items():
        if isinstance(value, str):
            out[key] = value.format(**combo)
        elif isinstance(value, dict):
            out[key] = {k: (v.format(**combo) if isinstance(v, str) else v) for k, v in value.items()}
        else:
            out[key] = value
    return out


def _cross_product(axes: dict) -> list[dict]:
    """Cross-product of ``{axis: [values]}`` preserving declaration order."""
    combos: list[dict] = [{}]
    for axis, values in axes.items():
        combos = [{**combo, axis: value} for combo in combos for value in values]
    return combos


def _descriptor_for(target: str) -> Path:
    from merlin.common.paths import repo_root
    return (repo_root() / "merlin" / "experiments" / "capsule_bench" / "targets" / target
            / "target_experiment.yaml")


def _ensure_contract_on_path(descriptor: Path) -> None:
    """If the descriptor names an out-of-tree ``target_contract`` (e.g. radiance's contract lives under
    the ``radiance`` target package), prepend its package root to ``MERLIN_TARGET_PATH`` so the registry
    resolves the manifest. Read from the descriptor, so it stays target-agnostic."""
    from merlin.common.paths import repo_root
    raw = yaml.safe_load(descriptor.read_text())
    tc = (raw.get("hardware_spec") or {}).get("target_contract")
    if not tc:
        return
    pkg = (repo_root() / tc).resolve().parent.parent      # .../contracts/target_contract.yaml -> package root
    cur = os.environ.get("MERLIN_TARGET_PATH", "")
    if str(pkg) not in cur.split(os.pathsep):
        os.environ["MERLIN_TARGET_PATH"] = os.pathsep.join([str(pkg), cur]) if cur else str(pkg)


def generate_target(target: str) -> list[Path]:
    descriptor = _descriptor_for(target)
    _ensure_contract_on_path(descriptor)
    te = load_target_experiment(descriptor)
    profile = load_profile(target)
    binding = CS.derive_binding(te, profile.get("datapath", {}))
    out_root = Path(te.capsule_corpus).parent                 # target's corpus root, derived (no move)
    # `sweeps:` (if any) expand into the same flat entries `capsules:` holds, so
    # everything downstream — builders, goldens, coverage — is unchanged.
    facts = _performance_facts(target)
    _sweep_skips: list = []
    _runtime_blocked: list = []
    _performance_errors: list = []
    entries = expand_sweeps(
        profile, binding, trait_facts=facts, skipped=_sweep_skips,
        blocked_unimplemented=_runtime_blocked, errors=_performance_errors)
    entries = [_resolve_flat_extents(e, binding) for e in entries]
    for _s in _sweep_skips:
        print(f"  [skip] performance family {_s['family']}: gate {_s['gate']['outcome']}")
    template = copy.deepcopy(profile.get("_performance_template") or {})
    declared_families = [dict(row) for row in (template.get("families") or [])]
    family_counts = {
        row["family"]: {"admitted_members": 0, "written_members": 0}
        for row in declared_families
    }
    for entry in entries:
        family = (entry.get("performance") or {}).get("family")
        if family:
            family_counts.setdefault(family, {"admitted_members": 0, "written_members": 0})
            family_counts[family]["admitted_members"] += 1
    # SCRUB EACH CAPSULE AS IT IS WRITTEN, not after the whole corpus succeeds. Scrubbing at the end
    # means one unrelated failure -- a capture that needs an external exporter, say -- aborts the run
    # with every capsule written so far still carrying its absolute `prov.weights_file` path. Measured:
    # a run that died on the last entry left `/scratch/.../capsule_m2m_<rand>/weights.safetensors` in
    # tracked MLIR across six capsules, in a repo that is published. Hygiene that only holds on the
    # happy path is not hygiene.
    #
    # ONE FAILING CAPSULE MUST NOT DESTROY THE WHOLE CORPUS. Letting the exception propagate meant a
    # single entry that needs an external exporter took every entry after it down with it: measured, a
    # capture that torch.export refuses (an LSTM whose `_flat_weights` are assigned rather than
    # registered) aborted the run before any of the tail-path sweep capsules were written, so a coverage
    # gap stayed open for a reason that had nothing to do with it. Failures are COLLECTED, reported by
    # name, and re-raised at the end -- the run still fails, it just fails after doing the work it could.
    written, failures = [], []
    for e in entries:
        family = (e.get("performance") or {}).get("family")
        try:
            w = _write_capsule(e, binding, out_root)
        except Exception as exc:                              # noqa: BLE001 — reported, never swallowed
            detail = f"{type(exc).__name__}: {str(exc)[:300]}"
            failures.append((e.get("name", "?"), detail))
            if family:
                _performance_errors.append({
                    "family": family, "member": e.get("name", "?"), "status": "error",
                    "error_type": type(exc).__name__, "detail": str(exc)[:500],
                })
            continue
        if w:
            _scrub_capsule_dir(w)
            written.append(w)
            if family:
                family_counts[family]["written_members"] += 1
        elif family:
            _performance_errors.append({
                "family": family, "member": e.get("name", "?"), "status": "error",
                "error_type": "NoOutput", "detail": "capsule writer returned no output",
            })
    if failures:
        print(f"  [FAIL] {len(failures)} capsule(s) could not be written:")
        for name, why in failures:
            print(f"    - {name}: {why}")
    # Record provenance for what we just emitted. The MANIFEST header has always CLAIMED the generator
    # rewrites it, but no writer existed, so it drifted silently as soon as the corpus grew.
    declared_blocked = [{
        "family": row["family"],
        "status": "blocked_unimplemented",
        "reason": row["reason"],
        "emitter": copy.deepcopy(row["performance"]["emitter"]),
        "fit_axes": list(row.get("fit_axes") or []),
        "comparison_roles": list(row.get("comparison_roles") or []),
    } for row in (template.get("blocked_unimplemented") or [])]
    generated_members = sum(row["written_members"] for row in family_counts.values())
    performance_record = {
        "shared_template": {"path": template.get("path"), "sha256": template.get("sha256")},
        "facts": {"target": target, "sha256": facts["sha256"]},
        "phase": {
            "category": "_perf",
            "label": "dev",
            "included_in_functional_grade": False,
            "exclusion": "TargetExperiment.corpus_siblings excludes underscore-prefixed categories",
        },
        "families": declared_families,
        "counts": {
            "declared_families": len(declared_families),
            "generated_families": sum(1 for row in family_counts.values()
                                      if row["written_members"] > 0),
            "generated_members": generated_members,
            "by_family": family_counts,
        },
        "skipped_inapplicable": _sweep_skips,
        "blocked_unimplemented": declared_blocked + _runtime_blocked,
        "errors": _performance_errors,
    }
    update_provenance_manifest(written, target=target, performance_record=performance_record)
    if failures:
        raise RuntimeError(
            f"{len(failures)} capsule(s) failed to generate: {', '.join(n for n, _ in failures)}; the "
            f"rest of the corpus was written, so re-running after fixing them is cheap")
    return written


def update_provenance_manifest(written, cap_root=None, *, target: str | None = None,
                               performance_record: "dict | None" = None) -> Path:
    """Rewrite ``MANIFEST.yaml``'s generated/hand_authored split from what this run actually emitted.

    The file's own header has always claimed the generator rewrites it, but no writer existed, so it was
    hand-maintained and silently drifted the moment the corpus grew -- 19 capsules appeared on disk that
    it never listed, and the only thing that noticed was a test telling you to "re-run generate_corpus.py",
    which did not do it.

    MERGE, never replace. A path this run emitted is ``generated``; everything else on disk keeps whatever
    classification it already had, defaulting to ``hand_authored`` for a capsule with no generator. That
    ordering matters: rebuilding the split from scratch would reclassify the frozen hand-authored
    source-of-record (A1, B3/B4, the held-out hidden set) as generated the first time a run happened to
    emit something at the same path.

    Scoped to the SHARED corpus (``<category>/<capsule>``, rel-depth 2). A target with its own nested
    corpus (``atlas/<category>/<capsule>``) carries its own provenance and is deliberately untouched.

    HOLDOUTS ARE COUNTED, NEVER NAMED. This file is tracked and sits inside the ``merlin/contract/``
    tree every arm is granted read-only, so listing a ``hidden/<capsule>`` path told the agent under
    test the op family of a held-out capsule. The generated/hand_authored split is provenance about
    the PUBLIC corpus; the holdouts contribute only a count, which reveals nothing.
    """
    root = Path(cap_root) if cap_root else Path(__file__).resolve().parent
    man_path = root / "MANIFEST.yaml"
    man = yaml.safe_load(man_path.read_text(encoding="utf-8")) if man_path.is_file() else {}
    gen, hand = set(man.get("generated") or []), set(man.get("hand_authored") or [])

    def _rel(d):
        try:
            r = Path(d).resolve().relative_to(root)
        except ValueError:
            return None
        return str(r) if len(r.parts) == 2 else None

    def _held(rel: str) -> bool:
        return rel.split("/", 1)[0] == "hidden"

    for d in written or []:
        rel = _rel(d)
        if rel:
            gen.add(rel); hand.discard(rel)
    on_disk = {str(rel) for c in root.rglob("capsule.yaml")
               if len((rel := c.parent.relative_to(root)).parts) == 2}
    hand |= (on_disk - gen - hand)          # never seen by a generator -> frozen source-of-record
    gen &= on_disk; hand &= on_disk         # drop entries whose capsule is gone

    # Split the holdouts back out: they are counted here, never named (see the docstring).
    held_gen, held_hand = {r for r in gen if _held(r)}, {r for r in hand if _held(r)}
    gen -= held_gen; hand -= held_hand

    man["generated_by"] = "merlin/contract/capsules/generate_corpus.py"
    man["generated"] = sorted(gen)
    man["hand_authored"] = sorted(hand)
    man["held_out"] = {"n_generated": len(held_gen), "n_hand_authored": len(held_hand)}
    if target is not None:
        per_target = dict(man.get("performance_generation") or {})
        per_target[target] = copy.deepcopy(performance_record or {})
        man["performance_generation"] = per_target
    head = man_path.read_text(encoding="utf-8").split("generated_by:")[0] if man_path.is_file() else ""
    man_path.write_text(head + yaml.safe_dump(man, sort_keys=False), encoding="utf-8")
    return man_path


def build_comparison_manifest(targets: list[str]) -> dict:
    """Group capsules that exercise the SAME op across targets into comparison sets, so a shared op (e.g.
    rmsnorm/gelu/gemv_batched) can be compared across each target's own precision (MXFP8 on mx vs FP8-E4M3
    on atlas vs fp16 on radiance). Keyed by ``comparison_group`` when the profile declares one, else by op."""
    groups: dict[str, list[dict]] = {}
    for t in targets:
        # public half only: this manifest is a published artifact, and one that names the holdouts
        # leaks exactly what splitting them out of the profile was meant to stop.
        prof = load_profile(t, include_holdouts=False)
        for e in prof["capsules"]:
            if e.get("kind") == "model" or e.get("op") == "model":
                continue
            key = e.get("comparison_group") or e.get("op", "unknown")
            groups.setdefault(key, []).append(
                {"target": t, "name": e["name"],
                 "dtype": e.get("operand_dtype", prof.get("datapath", {}).get("operand_dtype", "")),
                 "label": e.get("label", "public")})
    # a comparison set is only interesting when >1 target covers the op
    cross = {k: v for k, v in sorted(groups.items()) if len({m["target"] for m in v}) > 1}
    return {"comparison_sets": cross,
            "note": "each set is one op exercised across multiple targets in each target's own precision; "
                    "same inner op name across targets makes target-vs-target numerics directly comparable"}


def write_comparison_manifest(targets: list[str]) -> Path:
    from merlin.common.paths import artifacts_dir
    manifest = build_comparison_manifest(targets)
    out = Path(artifacts_dir()) / "compare" / "capsule_comparison_manifest.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Unified descriptor-driven capsule-corpus generator.")
    ap.add_argument("--target", default=None, help="one target (default: every target with a profile)")
    ap.add_argument("--comparison-manifest", action="store_true",
                    help="also emit the cross-target op-comparison manifest under out/artifacts/compare/")
    a = ap.parse_args(argv)
    targets = [a.target] if a.target else profile_targets()
    for t in targets:
        written = generate_target(t)
        print(f"{t}: wrote {len(written)} capsules -> {written[0].parent.parent if written else '(none)'}")
    if a.comparison_manifest or not a.target:
        allt = profile_targets()
        m = write_comparison_manifest(allt)
        print(f"comparison manifest: {m} ({len(build_comparison_manifest(allt)['comparison_sets'])} sets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
