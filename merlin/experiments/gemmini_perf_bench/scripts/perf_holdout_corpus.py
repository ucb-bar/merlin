#!/usr/bin/env python3
"""Commit/reveal construction of the generated performance holdout corpus.

The performance agent may see :func:`commit_holdout`'s public JSON, but it must
not see the seed, selected members, or generated operands.  Those live in a
separate host-only directory and are revealed only after every predeclared
candidate record proves that its compiler tree is sealed and immutable.

This module deliberately does not contain a second workload generator.  It
selects both unseen K-only predictor points and a separate unseen M/N/K
generalization cohort from the shared ``PK`` contract, then hands every exact
point back to ``contract/capsules/generate_corpus.py`` and ``corpus_spec`` for
capsule, operand, and golden generation.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = 2
ALGORITHM = "sha256-ranked-structural-domain-without-replacement"
ALGORITHM_VERSION = "perf-holdout-v2"
SEED_BYTES = 32
MIN_MEMBERS = 4
GENERALIZATION_FAMILY = "PKG"

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
GENERATOR = REPO / "merlin" / "contract" / "capsules" / "generate_corpus.py"
CORPUS_SPEC = REPO / "merlin" / "python" / "merlin" / "targetgen" / "corpus_spec.py"


class HoldoutError(RuntimeError):
    """The holdout boundary or its frozen evidence failed closed."""


@dataclass(frozen=True)
class HoldoutPaths:
    public_commitment: Path
    host_private_dir: Path
    seed: Path
    state: Path


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True) + "\n").encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _assert_plain_file(path: Path, *, label: str) -> Path:
    path = Path(path)
    if (path.is_symlink() or any(parent.is_symlink() for parent in path.parents)
            or not path.is_file()):
        raise HoldoutError(f"{label} is absent or is not a plain file: {path}")
    return path.resolve()


def _assert_frozen_file(path: Path, *, label: str) -> Path:
    path = _assert_plain_file(path, label=label)
    if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise HoldoutError(f"{label} is writable rather than frozen: {path}")
    return path


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _fresh_parent(path: Path, *, label: str) -> Path:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise HoldoutError(f"{label} is not fresh: {path}")
    parent = path.parent
    if (parent.is_symlink() or any(ancestor.is_symlink() for ancestor in parent.parents)
            or not parent.is_dir()):
        raise HoldoutError(f"{label} parent is absent or linked: {parent}")
    return path


def _write_exclusive(path: Path, payload: bytes, mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    path.chmod(mode)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    path = _assert_plain_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HoldoutError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise HoldoutError(f"{label} must contain a JSON object")
    return value


def _dtype_bytes(token: str) -> int:
    aliases = {"int8": 1, "i8": 1, "uint8": 1, "u8": 1,
               "int32": 4, "i32": 4, "uint32": 4, "u32": 4}
    try:
        return aliases[str(token).lower()]
    except KeyError as exc:
        raise HoldoutError(f"PK holdout cannot derive byte width for RTL dtype {token!r}") from exc


def _one_named(rows: Sequence[Mapping[str, Any]], name: str, *, label: str) -> Mapping[str, Any]:
    found = [row for row in rows if isinstance(row, Mapping) and row.get("name") == name]
    if len(found) != 1:
        raise HoldoutError(f"RTL/CIRCT facts need exactly one {label} named {name!r}")
    return found[0]


def _resolve_extent(token: object, tile: int) -> int:
    """Load the canonical extent resolver rather than carrying another parser."""
    generator = _load_generator()
    try:
        return int(generator.resolve_extent(token, tile))
    except Exception as exc:  # noqa: BLE001 - convert generator rejection to boundary refusal
        raise HoldoutError(f"shared PK extent {token!r} is invalid at tile={tile}: {exc}") from exc


def _pk_sweep(profile_path: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise HoldoutError(f"shared performance profile is unreadable: {profile_path}") from exc
    sweeps = [row for row in (document.get("sweeps") or [])
              if isinstance(row, dict) and row.get("id") == "PK"]
    if len(sweeps) != 1:
        raise HoldoutError("shared performance profile needs exactly one PK sweep")
    sweep = copy.deepcopy(sweeps[0])
    base = sweep.get("base") or {}
    performance = base.get("performance") or {}
    if (base.get("op") != "matmul" or performance.get("family") != "PK"
            or performance.get("claim") != "PREDICTS"
            or (performance.get("emitter") or {}).get("entry")
            != "merlin.targetgen.corpus_spec.build"):
        raise HoldoutError("shared PK family is not the runnable corpus_spec matmul contract")
    return sweep


def _source_paths(target: str, rtl_facts: Path, perf_profile: Path) -> dict[str, Path]:
    paths = {
        "rtl_circt_facts": rtl_facts,
        "shared_perf_contract": perf_profile,
        "generate_corpus_py": GENERATOR,
        "corpus_spec_py": CORPUS_SPEC,
    }
    descriptor = (REPO / "merlin" / "experiments" / "capsule_bench" / "targets"
                  / target / "target_experiment.yaml")
    public_profile = REPO / "merlin" / "contract" / "capsules" / "profiles" / f"{target}.yaml"
    if descriptor.is_file() and not descriptor.is_symlink():
        paths["target_experiment"] = descriptor
        raw = yaml.safe_load(descriptor.read_text(encoding="utf-8")) or {}
        contract = (raw.get("hardware_spec") or {}).get("target_contract")
        if contract:
            contract_path = (REPO / str(contract)).resolve()
            if contract_path.is_file() and not contract_path.is_symlink():
                paths["target_contract"] = contract_path
    if public_profile.is_file() and not public_profile.is_symlink():
        paths["target_public_profile"] = public_profile
    for label, path in paths.items():
        _assert_plain_file(path, label=label)
    return paths


def verify_rtl_facts_provenance(
        rtl_facts_path: Path, *, target: str,
        fact_builder: Any = None, core_hw_resolver: Any = None,
        extractor_path: Path | None = None) -> dict[str, str]:
    """Replay the CIRCT extractor and bind its actual RTL inputs before authoring.

    A generator name inside JSON is not provenance.  This check hashes the currently executing
    extractor, the resolved CIRCT HW dialect, and the source FIRRTL, then reruns extraction and
    requires the complete generated document to match the frozen snapshot byte-for-value.  The
    injected seams exist only for offline tests; production callers use the repository extractor.
    """
    facts_path = _assert_frozen_file(rtl_facts_path, label="frozen RTL/CIRCT facts")
    recorded = _load_json(facts_path, label="frozen RTL/CIRCT facts")
    generator = recorded.get("generator") or {}
    inputs = recorded.get("inputs") or {}
    facts = recorded.get("facts") or {}
    if (generator.get("name") != "merlin.targetgen.rtl.circt_introspect"
            or facts.get("target") != target or inputs.get("target") != target):
        raise HoldoutError("RTL facts do not name the exact CIRCT extractor and target")

    if fact_builder is None or core_hw_resolver is None:
        from merlin.targetgen.rtl import circt_introspect, mlc_bridge
        fact_builder = fact_builder or circt_introspect.build_facts
        core_hw_resolver = core_hw_resolver or mlc_bridge.core_hw_mlir
        extractor_path = extractor_path or Path(circt_introspect.__file__)
    extractor = _assert_plain_file(
        extractor_path or (REPO / "merlin/python/merlin/targetgen/rtl/circt_introspect.py"),
        label="CIRCT facts extractor")
    extractor_sha = _sha256_file(extractor)
    recorded_extractor = inputs.get("extractor_sha")
    if (not isinstance(recorded_extractor, str) or len(recorded_extractor) < 16
            or extractor_sha[:len(recorded_extractor)] != recorded_extractor):
        raise HoldoutError("frozen RTL facts were produced by a different extractor revision")

    core_path = core_hw_resolver(target)
    if core_path is None:
        raise HoldoutError("CIRCT core HW dialect cannot be resolved for provenance replay")
    core_path = _assert_plain_file(Path(core_path), label="CIRCT core HW dialect")
    core_sha = _sha256_file(core_path)
    if inputs.get("core_hw_sha256") != core_sha:
        raise HoldoutError("frozen RTL facts do not bind the resolved CIRCT core HW dialect")

    source = facts.get("source") or {}
    fir_path = _assert_plain_file(Path(str(source.get("fir_path") or "")), label="source FIRRTL")
    fir_sha = _sha256_file(fir_path)
    recorded_fir = inputs.get("fir_sha")
    if (not isinstance(recorded_fir, str) or len(recorded_fir) < 16
            or fir_sha[:len(recorded_fir)] != recorded_fir):
        raise HoldoutError("frozen RTL facts do not bind their source FIRRTL")

    try:
        replayed = fact_builder(target=target)
    except Exception as exc:  # noqa: BLE001 - toolchain failures are an admission refusal
        raise HoldoutError(
            f"CIRCT RTL facts could not be replayed ({type(exc).__name__})") from exc
    if replayed != recorded:
        raise HoldoutError("live CIRCT extraction differs from the frozen RTL facts snapshot")
    return {
        "rtl_facts_sha256": _sha256_file(facts_path),
        "extractor_sha256": extractor_sha,
        "core_hw_sha256": core_sha,
        "firrtl_sha256": fir_sha,
        "replay_sha256": _sha256(_canonical_json(replayed)),
    }


def derive_domain(rtl_facts_path: Path, perf_profile_path: Path, *, target: str) -> dict[str, Any]:
    """Derive PK predictor and M/N/K generalization domains from frozen inputs.

    The largest public PK point is the cost envelope: a hidden point may vary K
    within it, but may not demand more MACs or operand bytes.  RTL memory facts
    independently prove every point fits the operand and accumulator stores.
    Thus neither a previous timing result nor a hand-picked hidden shape enters
    selection.  A second domain uses mesh boundaries and adjacent tails on all
    three axes, subject to the same envelope, so it probes shape transfer
    without changing operation semantics.
    """
    rtl_facts_path = _assert_frozen_file(rtl_facts_path, label="frozen RTL/CIRCT facts")
    perf_profile_path = _assert_frozen_file(
        perf_profile_path, label="frozen shared performance profile")
    facts_document = _load_json(rtl_facts_path, label="frozen RTL/CIRCT facts")
    generator = facts_document.get("generator") or {}
    if not str(generator.get("name") or "").startswith(
            "merlin.targetgen.rtl.circt_introspect"):
        raise HoldoutError("facts were not produced by the CIRCT RTL introspector")
    facts = facts_document.get("facts") or {}
    if facts.get("target") != target:
        raise HoldoutError(
            f"RTL/CIRCT facts target {facts.get('target')!r} does not match {target!r}")
    mesh = _one_named(facts.get("arrays") or [], "mesh", label="array")
    rows, cols = int(mesh.get("rows") or 0), int(mesh.get("cols") or 0)
    if rows < 1 or cols < 1 or rows != cols:
        raise HoldoutError("shared PK binding needs a positive square RTL-derived mesh")
    scratchpad = _one_named(facts.get("memories") or [], "scratchpad", label="memory")
    accumulator = _one_named(facts.get("memories") or [], "accumulator", label="memory")
    scratchpad_bytes = int(scratchpad.get("bytes") or 0)
    accumulator_bytes = int(accumulator.get("bytes") or 0)
    datapaths = facts.get("datapaths") or []
    operand = _one_named(datapaths, "input", label="datapath")
    accum = _one_named(datapaths, "accumulator", label="datapath")
    operand_dtype, accum_dtype = str(operand.get("dtype") or ""), str(accum.get("dtype") or "")
    operand_bytes, accum_bytes = _dtype_bytes(operand_dtype), _dtype_bytes(accum_dtype)

    sweep = _pk_sweep(perf_profile_path)
    axes = sweep.get("axes") or {}
    if set(axes) != {"M", "N", "K"} or len(axes["M"]) != 1 or len(axes["N"]) != 1:
        raise HoldoutError("shared PK family must fix M/N and vary only K")
    m = _resolve_extent(axes["M"][0], rows)
    n = _resolve_extent(axes["N"][0], rows)
    if (m, n) != (rows, cols):
        raise HoldoutError("shared PK fixed axes drifted from the RTL-derived mesh tile")
    dev_k = sorted({_resolve_extent(token, rows) for token in axes["K"]})
    if len(dev_k) < 2:
        raise HoldoutError("shared PK family has fewer than two public K points")
    maximum_k = max(dev_k)
    acc_working_bytes = m * n * accum_bytes
    if acc_working_bytes > accumulator_bytes:
        raise HoldoutError("the public PK tile does not fit the RTL-derived accumulator capacity")

    legal: list[int] = []
    for k in range(min(dev_k), maximum_k + 1):
        operand_working_bytes = (m * k + k * n) * operand_bytes
        macs = m * n * k
        if (k not in dev_k and operand_working_bytes <= scratchpad_bytes
                and acc_working_bytes <= accumulator_bytes
                and macs <= m * n * maximum_k):
            legal.append(k)
    if len(legal) < MIN_MEMBERS:
        raise HoldoutError(
            f"RTL-derived PK domain has only {len(legal)} unseen legal point(s); need {MIN_MEMBERS}")

    sources = _source_paths(target, rtl_facts_path, perf_profile_path)
    positive_depths = [int(row["pipeline_depth"])
                       for row in (facts.get("timing") or [])
                       if isinstance(row, Mapping)
                       and isinstance(row.get("pipeline_depth"), int)
                       and row["pipeline_depth"] > 0]
    if not positive_depths:
        raise HoldoutError("RTL/CIRCT facts do not establish structural pipeline depth for PK")
    maximum_macs = m * n * maximum_k
    maximum_operand_bytes = (m * maximum_k + maximum_k * n) * operand_bytes
    maximum_accumulator_bytes = m * n * accum_bytes

    # These are structural landmarks, not target-specific shapes.  For every
    # mesh boundary admitted by the public coordinate envelope, test the exact
    # boundary and its two tails.  The upper bound is itself taken from the
    # largest public axis extent.  This catches both tiling and remainder bugs
    # without embedding a Gemmini dimension or consulting timing results.
    maximum_extent = max(m, n, maximum_k)
    landmarks = sorted({value
                        for boundary in range(rows, maximum_extent + 1, rows)
                        for value in (boundary - 1, boundary, boundary + 1)
                        if 1 <= value <= maximum_extent}
                       | {1, maximum_extent})
    public_shapes = {(m, n, k) for k in dev_k}
    legal_shapes = []
    for gm in landmarks:
        for gn in landmarks:
            for gk in landmarks:
                operand_working_bytes = (gm * gk + gk * gn) * operand_bytes
                accumulator_working_bytes = gm * gn * accum_bytes
                if ((gm, gn, gk) not in public_shapes
                        and gm * gn * gk <= maximum_macs
                        and operand_working_bytes <= maximum_operand_bytes
                        and accumulator_working_bytes <= maximum_accumulator_bytes
                        and operand_working_bytes <= scratchpad_bytes
                        and accumulator_working_bytes <= accumulator_bytes):
                    legal_shapes.append((gm, gn, gk))
    if len(legal_shapes) < MIN_MEMBERS:
        raise HoldoutError(
            "RTL/public-contract-derived M/N/K domain has too few legal unseen shapes")
    if any(len({shape[index] for shape in legal_shapes}) < 2 for index in range(3)):
        raise HoldoutError("M/N/K generalization domain does not vary all three axes")

    return {
        "target": target,
        "family": "PK",
        "operation": "matmul",
        "varied_axis": "K",
        "fixed_shape": {"M": m, "N": n},
        "legal_k": {"minimum": min(legal), "maximum": max(legal),
                    "cardinality": len(legal), "excluded_public_dev": dev_k},
        "bounds": {
            "mesh": {"rows": rows, "cols": cols},
            "scratchpad_bytes": scratchpad_bytes,
            "accumulator_bytes": accumulator_bytes,
            "operand_dtype": operand_dtype,
            "operand_element_bytes": operand_bytes,
            "accumulator_dtype": accum_dtype,
            "accumulator_element_bytes": accum_bytes,
            "cost_envelope": {
                "basis": "largest_public_PK_member_from_shared_contract",
                "maximum_K": maximum_k,
                "maximum_coordinate_extent": maximum_extent,
                "maximum_macs": maximum_macs,
                "maximum_operand_bytes": maximum_operand_bytes,
                "maximum_accumulator_bytes": maximum_accumulator_bytes,
            },
            "rtl_positive_pipeline_depths": sorted(set(positive_depths)),
        },
        "generalization": {
            "family": GENERALIZATION_FAMILY,
            "source_family": "PK",
            "claim": "DIFFERENTIAL",
            "claim_scope": (
                "paired whole-compiler speedup generalization over unseen legal matmul M/N/K "
                "shapes; not a new kernel semantic or a roofline claim"),
            "semantic_scope": {
                "operation": "matmul",
                "epilogue": "none",
                "reason": (
                    "the frozen admitted PK source contract exposes matmul through corpus_spec "
                    "and declares no runnable semantic or epilogue variants"),
            },
            "selection_domain": {
                "axis_landmark_algorithm": "mesh_multiples_and_adjacent_tails_within_public_extent",
                "axis_landmarks": landmarks,
                "excluded_public_shapes": [
                    {"M": pm, "N": pn, "K": pk}
                    for pm, pn, pk in sorted(public_shapes)
                ],
                "cardinality": len(legal_shapes),
                "coverage_requirement": {
                    "all_axes": "at_least_two_distinct_values_on_each_of_M_N_K",
                    "two_dimensional": "at_least_one_member_has_M_gt_1_and_N_gt_1",
                    "M_mesh_boundary": "values_include_at_most_mesh_rows_and_above_mesh_rows",
                    "N_mesh_boundary": "values_include_at_most_mesh_cols_and_above_mesh_cols",
                    "tails": "M_and_N_each_include_a_non_mesh-multiple_extent",
                },
            },
        },
        "source_sha256": {label: _sha256_file(path) for label, path in sorted(sources.items())},
    }


def _legal_values(domain: Mapping[str, Any]) -> list[int]:
    legal = domain.get("legal_k") or {}
    start, stop = int(legal.get("minimum") or 0), int(legal.get("maximum") or 0)
    excluded = {int(value) for value in (legal.get("excluded_public_dev") or [])}
    values = [value for value in range(start, stop + 1) if value not in excluded]
    if len(values) != legal.get("cardinality"):
        raise HoldoutError("committed PK domain cardinality is internally inconsistent")
    return values


def select_members(seed: bytes, domain: Mapping[str, Any], count: int) -> list[int]:
    """Stable seeded selection without depending on a language PRNG version."""
    if not isinstance(seed, bytes) or len(seed) != SEED_BYTES:
        raise HoldoutError(f"holdout seed must contain exactly {SEED_BYTES} bytes")
    if isinstance(count, bool) or not isinstance(count, int) or count < MIN_MEMBERS:
        raise HoldoutError(f"holdout needs at least {MIN_MEMBERS} members")
    values = _legal_values(domain)
    if count > len(values):
        raise HoldoutError(f"holdout count {count} exceeds legal domain size {len(values)}")
    domain_digest = hashlib.sha256(_canonical_json(domain)).digest()
    ranked = sorted(values, key=lambda value: (
        hashlib.sha256(seed + b"\0" + domain_digest + value.to_bytes(8, "big")).digest(), value))
    return sorted(ranked[:count])


def _legal_generalization_shapes(domain: Mapping[str, Any]) -> list[tuple[int, int, int]]:
    generalization = domain.get("generalization") or {}
    selection = generalization.get("selection_domain") or {}
    landmarks = [int(value) for value in (selection.get("axis_landmarks") or [])]
    excluded = {(int(row["M"]), int(row["N"]), int(row["K"]))
                for row in (selection.get("excluded_public_shapes") or [])}
    bounds = domain.get("bounds") or {}
    cost = bounds.get("cost_envelope") or {}
    operand_bytes = int(bounds.get("operand_element_bytes") or 0)
    accum_bytes = int(bounds.get("accumulator_element_bytes") or 0)
    scratchpad_bytes = int(bounds.get("scratchpad_bytes") or 0)
    accumulator_bytes = int(bounds.get("accumulator_bytes") or 0)
    legal = []
    for m in landmarks:
        for n in landmarks:
            for k in landmarks:
                operand_working = (m * k + k * n) * operand_bytes
                accumulator_working = m * n * accum_bytes
                if ((m, n, k) not in excluded
                        and m * n * k <= int(cost.get("maximum_macs") or 0)
                        and operand_working <= int(cost.get("maximum_operand_bytes") or 0)
                        and accumulator_working <= int(
                            cost.get("maximum_accumulator_bytes") or 0)
                        and operand_working <= scratchpad_bytes
                        and accumulator_working <= accumulator_bytes):
                    legal.append((m, n, k))
    if len(legal) != selection.get("cardinality"):
        raise HoldoutError("committed M/N/K domain cardinality is internally inconsistent")
    return legal


def select_generalization_members(
        seed: bytes, domain: Mapping[str, Any], count: int) -> list[dict[str, int]]:
    """Seed-rank exact unseen shapes while guaranteeing M/N/K coverage.

    Coverage is a precommitted property of the cohort, not a post-result
    filter.  Ranking uses a distinct domain separator from the PK predictor
    selection so the two cohorts cannot accidentally share a PRNG stream.
    """
    if not isinstance(seed, bytes) or len(seed) != SEED_BYTES:
        raise HoldoutError(f"holdout seed must contain exactly {SEED_BYTES} bytes")
    if isinstance(count, bool) or not isinstance(count, int) or count < MIN_MEMBERS:
        raise HoldoutError(f"generalization holdout needs at least {MIN_MEMBERS} members")
    values = _legal_generalization_shapes(domain)
    if count > len(values):
        raise HoldoutError(
            f"generalization count {count} exceeds legal domain size {len(values)}")
    digest = hashlib.sha256(_canonical_json(domain)).digest()
    ranked = sorted(values, key=lambda shape: (
        hashlib.sha256(seed + b"\0mnk\0" + digest
                       + b"".join(value.to_bytes(8, "big") for value in shape)).digest(),
        shape))
    mesh = (domain.get("bounds") or {}).get("mesh") or {}
    rows, cols = int(mesh.get("rows") or 0), int(mesh.get("cols") or 0)

    def coverage(shapes: Sequence[tuple[int, int, int]]) -> tuple[bool, ...]:
        m_values, n_values, k_values = ({shape[i] for shape in shapes} for i in range(3))
        return (
            len(m_values) >= 2, len(n_values) >= 2, len(k_values) >= 2,
            any(m > 1 and n > 1 for m, n, _ in shapes),
            any(m <= rows for m in m_values), any(m > rows for m in m_values),
            any(n <= cols for n in n_values), any(n > cols for n in n_values),
            any(m % rows for m in m_values), any(n % cols for n in n_values),
        )

    # Seed rank resolves every tie, but admission of a point is driven first by
    # how many still-missing precommitted coverage obligations it satisfies.
    # This avoids a statistically possible all-skinny random cohort while
    # retaining result-independent seeded selection.
    selected: list[tuple[int, int, int]] = []
    remaining = list(ranked)
    while len(selected) < count:
        before = sum(coverage(selected))
        best = max(
            enumerate(remaining),
            key=lambda indexed: (
                sum(coverage([*selected, indexed[1]])) - before,
                -indexed[0]),
        )
        selected.append(best[1])
        remaining.pop(best[0])
    if len(selected) != count or not all(coverage(selected)):
        raise HoldoutError(
            "seeded cohort cannot satisfy precommitted M/N/K mesh-boundary coverage")
    return [{"M": m, "N": n, "K": k} for m, n, k in sorted(selected)]


def commit_holdout(
        public_commitment: Path, host_private_dir: Path, *, rtl_facts_path: Path,
        perf_profile_path: Path, target: str, candidate_ids: Sequence[str], count: int = 4,
        generalization_count: int = 4,
        seed: bytes | None = None, prior_result_paths: Sequence[Path] = (),
        agent_view_root: Path | None = None) -> HoldoutPaths:
    """Freeze a public commitment and host-only seed/state before authoring."""
    if prior_result_paths:
        raise HoldoutError("holdout selection refuses every prior-result input")
    if isinstance(count, bool) or not isinstance(count, int) or count < MIN_MEMBERS:
        raise HoldoutError(f"holdout needs at least {MIN_MEMBERS} members")
    if (isinstance(generalization_count, bool)
            or not isinstance(generalization_count, int)
            or generalization_count < MIN_MEMBERS):
        raise HoldoutError(
            f"generalization holdout needs at least {MIN_MEMBERS} members")
    ids = [str(value) for value in candidate_ids]
    if not ids or any(not value or "/" in value or value in {".", ".."} for value in ids):
        raise HoldoutError("candidate ids must be non-empty safe path components")
    if len(set(ids)) != len(ids):
        raise HoldoutError("candidate ids must be unique")
    public_commitment = _fresh_parent(Path(public_commitment), label="public commitment")
    host_private_dir = _fresh_parent(Path(host_private_dir), label="host-private holdout directory")
    if agent_view_root is not None:
        agent_view_root = Path(agent_view_root).resolve(strict=False)
        if _inside(host_private_dir, agent_view_root):
            raise HoldoutError("host-private holdout directory is inside the agent view")
        if not _inside(public_commitment, agent_view_root):
            raise HoldoutError("public commitment is not inside the declared agent view")

    domain = {
        **derive_domain(rtl_facts_path, perf_profile_path, target=target),
        "expected_candidate_count": len(ids),
    }
    seed = os.urandom(SEED_BYTES) if seed is None else seed
    selected = select_members(seed, domain, count)
    selected_generalization = select_generalization_members(
        seed, domain, generalization_count)
    public = {
        "algorithm": ALGORITHM,
        "version": ALGORITHM_VERSION,
        "domain": domain,
        "cohort_counts": {"PK_predictor": count,
                          "PK_MNK_generalization": generalization_count},
        "seed_sha256": _sha256(seed),
    }
    # Selected values and candidate identities are intentionally private.
    private = {
        "schema_version": SCHEMA_VERSION,
        "public_commitment_sha256": _sha256(_canonical_json(public)),
        "candidate_ids": ids,
        "selected_k": selected,
        "selected_generalization_shapes": selected_generalization,
        "source_paths": {label: str(path) for label, path in sorted(
            _source_paths(target, Path(rtl_facts_path).resolve(), Path(perf_profile_path).resolve()).items())},
        "agent_view_root": str(agent_view_root) if agent_view_root is not None else None,
    }

    host_private_dir.mkdir(mode=0o700)
    host_private_dir.chmod(0o700)
    seed_path = host_private_dir / "seed.bin"
    state_path = host_private_dir / "state.json"
    try:
        _write_exclusive(seed_path, seed, 0o600)
        _write_exclusive(state_path, _canonical_json(private), 0o600)
        _write_exclusive(public_commitment, _canonical_json(public), 0o444)
    except Exception:
        # Leave any host evidence in place for audit; never overwrite it on retry.
        raise
    return HoldoutPaths(public_commitment.resolve(), host_private_dir.resolve(),
                        seed_path.resolve(), state_path.resolve())


def _verify_source_state(public: Mapping[str, Any], private: Mapping[str, Any]) -> None:
    recorded = (public.get("domain") or {}).get("source_sha256") or {}
    paths = private.get("source_paths") or {}
    if set(recorded) != set(paths):
        raise HoldoutError("private source path set drifted from the public commitment")
    for label in sorted(recorded):
        path = _assert_plain_file(Path(paths[label]), label=f"committed source {label}")
        if _sha256_file(path) != recorded[label]:
            raise HoldoutError(f"committed source changed after authoring began: {label}")


def _verify_candidate_seals(
        expected_ids: Sequence[str], candidate_seals: Mapping[str, Path]) -> list[dict[str, str]]:
    if set(candidate_seals) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(candidate_seals))
        extra = sorted(set(candidate_seals) - set(expected_ids))
        raise HoldoutError(f"candidate seal set is incomplete or foreign (missing={missing}, extra={extra})")
    try:
        from merlin.benchharness import hash_tree
    except ImportError as exc:
        raise HoldoutError("merlin.benchharness.hash_tree is unavailable") from exc
    receipts: list[dict[str, str]] = []
    for candidate_id in expected_ids:
        path = _assert_frozen_file(Path(candidate_seals[candidate_id]),
                                   label=f"candidate seal {candidate_id}")
        document = _load_json(path, label=f"candidate seal {candidate_id}")
        candidate = document.get("candidate") or {}
        admission = document.get("admission") or {}
        if (document.get("state") != "sealed" or candidate.get("read_only") is not True
                or admission.get("consumable") is not True or not _is_sha256(candidate.get("sha256"))):
            raise HoldoutError(f"candidate {candidate_id} is not a consumable sealed compiler")
        tree = Path(str(candidate.get("path") or ""))
        if tree.is_symlink() or not tree.is_dir():
            raise HoldoutError(f"candidate {candidate_id} sealed tree is absent or linked")
        for member in (tree, *tree.rglob("*")):
            if member.is_symlink():
                raise HoldoutError(f"candidate {candidate_id} tree contains a symlink: {member}")
            if member.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
                raise HoldoutError(f"candidate {candidate_id} tree is writable: {member}")
        observed = str(hash_tree(tree).get("sha256") or "")
        if observed != candidate["sha256"]:
            raise HoldoutError(f"candidate {candidate_id} sealed tree digest changed")
        receipts.append({"candidate_id": candidate_id, "record_sha256": _sha256_file(path),
                         "candidate_sha256": observed})
    return receipts


def _load_generator():
    spec = importlib.util.spec_from_file_location("merlin_perf_holdout_generate_corpus", GENERATOR)
    if spec is None or spec.loader is None:
        raise HoldoutError(f"cannot load shared corpus generator: {GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _binding(generator, target: str, domain: Mapping[str, Any]):
    descriptor = generator._descriptor_for(target)
    generator._ensure_contract_on_path(descriptor)
    target_experiment = generator.load_target_experiment(descriptor)
    public_profile = generator.load_profile(target, include_holdouts=False)
    binding = generator.CS.derive_binding(target_experiment, public_profile.get("datapath", {}))
    bounds = domain["bounds"]
    if (binding.tile_dim != bounds["mesh"]["rows"]
            or str(binding.operand_dtype) not in {bounds["operand_dtype"], "int8"}
            or str(binding.accum_dtype) not in {bounds["accumulator_dtype"], "int32"}):
        raise HoldoutError("live corpus binding drifted from the committed RTL/CIRCT datapath")
    return binding


def _tree_record(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise HoldoutError(f"materialized holdout contains a symlink: {path}")
        if path.is_file():
            rows.append({"path": path.relative_to(root).as_posix(),
                         "bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    return {"files": rows, "sha256": _sha256(_canonical_json(rows))}


def _make_host_readonly(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            raise HoldoutError(f"cannot seal linked holdout member: {path}")
        path.chmod(0o500 if path.is_dir() else 0o400)
    root.chmod(0o500)


def reveal_and_materialize(
        public_commitment: Path, host_private_dir: Path, output_dir: Path, *,
        candidate_seals: Mapping[str, Path]) -> Path:
    """Verify the commitment and all candidate seals, then generate the corpus.

    Returns the immutable host-only ``holdout_manifest.json``.  The output path
    must be fresh and outside the declared agent view.
    """
    public_path = _assert_frozen_file(Path(public_commitment), label="public holdout commitment")
    private_dir = Path(host_private_dir)
    if private_dir.is_symlink() or not private_dir.is_dir():
        raise HoldoutError("host-private holdout directory is absent or linked")
    if stat.S_IMODE(private_dir.stat().st_mode) != 0o700:
        raise HoldoutError("host-private holdout directory permissions are not 0700")
    seed_path = _assert_plain_file(private_dir / "seed.bin", label="host-private holdout seed")
    state_path = _assert_plain_file(private_dir / "state.json", label="host-private holdout state")
    if stat.S_IMODE(seed_path.stat().st_mode) != 0o600 or stat.S_IMODE(state_path.stat().st_mode) != 0o600:
        raise HoldoutError("host-private seed/state permissions are not 0600")
    public = _load_json(public_path, label="public holdout commitment")
    if set(public) != {"algorithm", "version", "domain", "cohort_counts", "seed_sha256"}:
        raise HoldoutError("public commitment contains fields outside the non-leaking schema")
    if public.get("algorithm") != ALGORITHM or public.get("version") != ALGORITHM_VERSION:
        raise HoldoutError("public commitment uses an unsupported selection algorithm")
    private = _load_json(state_path, label="host-private holdout state")
    if private.get("public_commitment_sha256") != _sha256(_canonical_json(public)):
        raise HoldoutError("public holdout commitment changed after it was prepared")
    seed = seed_path.read_bytes()
    if len(seed) != SEED_BYTES or _sha256(seed) != public.get("seed_sha256"):
        raise HoldoutError("holdout seed does not open the public commitment")
    _verify_source_state(public, private)
    counts = public.get("cohort_counts") or {}
    if set(counts) != {"PK_predictor", "PK_MNK_generalization"}:
        raise HoldoutError("public commitment does not predeclare both holdout cohorts")
    selected = select_members(seed, public["domain"], counts["PK_predictor"])
    selected_generalization = select_generalization_members(
        seed, public["domain"], counts["PK_MNK_generalization"])
    if selected != private.get("selected_k"):
        raise HoldoutError("host-private selected members do not follow the committed algorithm")
    if selected_generalization != private.get("selected_generalization_shapes"):
        raise HoldoutError(
            "host-private generalization members do not follow the committed algorithm")
    dev = set(public["domain"]["legal_k"]["excluded_public_dev"])
    if len(selected) != len(set(selected)) or dev.intersection(selected):
        raise HoldoutError("selected holdout members collide with each other or the public PK set")
    candidate_receipts = _verify_candidate_seals(
        private.get("candidate_ids") or [], candidate_seals)

    output_dir = _fresh_parent(Path(output_dir), label="holdout corpus output")
    agent_view = private.get("agent_view_root")
    if agent_view and _inside(output_dir, Path(agent_view)):
        raise HoldoutError("holdout corpus output is inside the agent view")
    output_dir.mkdir(mode=0o700)
    output_dir.chmod(0o700)
    try:
        generator = _load_generator()
        binding = _binding(generator, str(public["domain"]["target"]), public["domain"])
        perf_profile = Path(private["source_paths"]["shared_perf_contract"])
        sweep = _pk_sweep(perf_profile)
        sweep["axes"] = {
            "M": [int(public["domain"]["fixed_shape"]["M"])],
            "N": [int(public["domain"]["fixed_shape"]["N"])],
            "K": selected,
        }
        sweep["name"] = "PKH{i:02d}_k{K}"
        sweep["source_reference"] = (
            "commit/reveal held-out points from the shared PK family; selected only after candidates sealed")
        sweep["base"]["label"] = "held_out"
        # source_role records HOW a capsule was constructed, and these are expanded from the very
        # same PK sweep as the public members -- so `derived_sweep` is the accurate role and the one
        # the capsule schema admits. Emitting "generated_seeded_holdout" here made every revealed
        # capsule schema-INVALID, which failed the run at the reveal step after all three candidates
        # had already sealed. What that spelling was carrying -- that these points were chosen only
        # after sealing -- is not lost: it is in `label` (held_out) and in `source_reference` below,
        # and no consumer ever keyed on the role.
        sweep["base"]["source_role"] = "derived_sweep"
        trait_facts = {"traits": {"structural_pipeline_depth": {
            "satisfied": True,
            "tier": "rtl_circt_derived",
            "evidence": "positive pipeline depth in the committed CIRCT RTL facts",
            "missing": [],
        }}}
        skipped: list = []
        blocked: list = []
        errors: list = []
        entries = generator.expand_sweeps(
            {"sweeps": [sweep]}, binding, trait_facts=trait_facts, skipped=skipped,
            blocked_unimplemented=blocked, errors=errors)
        if skipped or blocked or errors or len(entries) != counts["PK_predictor"]:
            raise HoldoutError(
                f"shared PK generator refused committed holdout (skipped={skipped}, "
                f"blocked={blocked}, errors={errors}, members={len(entries)})")
        predictor_written: list[Path] = []
        for entry in entries:
            path = generator._write_capsule(entry, binding, output_dir)
            if path is None:
                raise HoldoutError(f"shared generator skipped holdout member {entry.get('name')}")
            generator._scrub_capsule_dir(path)
            predictor_written.append(Path(path))
        predictor_names = [path.name for path in predictor_written]
        if (len(predictor_names) != len(set(predictor_names))
                or len(predictor_written) != len(selected)):
            raise HoldoutError("generated holdout member names collided")

        # Each exact generalization point is materialized from PK's admitted
        # base contract, then resolved by the same performance materializer and
        # corpus_spec builder as the public sweep.  We intentionally do not
        # create a tracked target-specific family or a parallel workload path.
        generalization_written: list[Path] = []
        for index, shape in enumerate(selected_generalization):
            entry = copy.deepcopy(sweep["base"])
            entry.update(shape)
            entry["name"] = (
                f"PKG{index:02d}_m{shape['M']}k{shape['K']}n{shape['N']}")
            entry["label"] = "held_out_generalization"
            entry["source_role"] = "derived_sweep"   # see the note at the PKH sweep above
            entry["source_reference"] = (
                "commit/reveal M/N/K generalization point from PK's runnable matmul contract; "
                "selected before authoring and revealed only after all candidates sealed")
            generator._materialize_performance_entry(entry, binding)
            path = generator._write_capsule(entry, binding, output_dir)
            if path is None:
                raise HoldoutError(
                    f"shared generator skipped generalization member {entry['name']}")
            generator._scrub_capsule_dir(path)
            generalization_written.append(Path(path))
        generalization_names = [path.name for path in generalization_written]
        all_names = predictor_names + generalization_names
        if (len(generalization_written) != len(selected_generalization)
                or len(all_names) != len(set(all_names))):
            raise HoldoutError("generated generalization member names collided")
        capsules = _tree_record(output_dir)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "generated_performance_holdout_reveal",
            "commitment": {"path": str(public_path), "sha256": _sha256_file(public_path),
                           "algorithm": ALGORITHM, "version": ALGORITHM_VERSION},
            "reveal": {"seed_hex": seed.hex(), "seed_sha256": _sha256(seed)},
            "domain": public["domain"],
            "cohorts": {
                "PK_predictor": {
                    "family": "PK", "source_family": "PK", "claim": "PREDICTS",
                    "varied_axes": ["K"], "member_count": len(predictor_names),
                },
                "PK_MNK_generalization": {
                    "family": GENERALIZATION_FAMILY, "source_family": "PK",
                    "claim": "DIFFERENTIAL", "varied_axes": ["M", "N", "K"],
                    "member_count": len(generalization_names),
                    "scope": public["domain"]["generalization"]["claim_scope"],
                    "descriptor_contract_family": "PK",
                    "identity_rule": (
                        "capsule descriptor retains the admitted PK builder contract; host reveal "
                        "manifest assigns PKG measurement identity and PK predictor analysis must "
                        "exclude these members"),
                },
            },
            "members": ([
                {"name": name, "path": f"_perf/{name}", "family": "PK",
                 "cohort": "PK_predictor",
                 "M": public["domain"]["fixed_shape"]["M"],
                 "N": public["domain"]["fixed_shape"]["N"], "K": k}
                for name, k in zip(predictor_names, selected, strict=True)
            ] + [
                {"name": name, "path": f"_perf/{name}",
                 "family": GENERALIZATION_FAMILY, "source_family": "PK",
                 "descriptor_contract_family": "PK",
                 "cohort": "PK_MNK_generalization", **shape}
                for name, shape in zip(
                    generalization_names, selected_generalization, strict=True)
            ]),
            "public_dev_k": sorted(dev),
            "candidate_seals": candidate_receipts,
            "corpus": capsules,
            "generator": {
                "entry": "merlin.contract.capsules.generate_corpus._write_capsule",
                "builder": "merlin.targetgen.corpus_spec.build",
                "operands": "capsule_golden deterministic operand synthesis keyed by generated capsule name",
            },
        }
        manifest_path = output_dir / "holdout_manifest.json"
        _write_exclusive(manifest_path, _canonical_json(manifest), 0o400)
        _make_host_readonly(output_dir)
        return manifest_path.resolve()
    except Exception:
        # A partial corpus is evidence of refusal, never silently reused: the
        # fresh-output gate prevents it from being mistaken for a later run.
        raise


__all__ = [
    "ALGORITHM", "ALGORITHM_VERSION", "HoldoutError", "HoldoutPaths",
    "commit_holdout", "derive_domain", "reveal_and_materialize", "select_members",
    "select_generalization_members",
]
