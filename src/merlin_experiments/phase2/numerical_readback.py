"""Host-private reconstruction of numerical agreement from two protected readbacks.

The caller is the trusted evaluator, supplying its original V4 provenance,
fixed budget and selected BuildOnlyService. These are existing ownership
capabilities, not candidate action fields. Freezing a console does not prove
that an executable ran, that a reference is scientifically valid, or that a
hardware configuration/timer was selected. This owner grants none of those
claims and never constructs a FinalMemberComparison.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from merlin.common.quant_formats import machine_bits
from merlin.perf.float_accuracy import compare
from merlin.perf.phase2_portfolio import QualityBudget, QualityObservation
from merlin.runtime.backends.base import decode_float_readback, parse_console
from merlin.runtime.commandbuffer import declared_output_dtypes
from merlin.runtime.fp8_formats import float_format_of, storage_bits
from merlin.targetgen.contract import readback_policy as RB
from merlin.targetgen.contract.build_service import BuildOnlyService
from merlin.targetgen.sandbox import bwrap as BW
from merlin_experiments.phase2.campaign import verify_private_input_snapshot
from merlin_experiments.phase2.contracts import StageGateError, sha256_file


@dataclass(frozen=True)
class ReadbackFiles:
    """Original evaluator-owned paths, resolved only through private V4 inputs."""

    command_buffer: Path
    build_receipt: Path
    kernel_object: Path
    harness: Path
    executable: Path
    console: Path

    def paths(self) -> tuple[Path, ...]:
        return (
            self.command_buffer,
            self.build_receipt,
            self.kernel_object,
            self.harness,
            self.executable,
            self.console,
            self.harness.parent / "out_b64.h",
        )


@dataclass(frozen=True)
class NumericalReadback:
    """Computed full-output observation; no execution or hardware authority."""

    quality: QualityObservation
    element_count: int
    outputs: tuple[str, ...]
    evidence: tuple[tuple[str, str], ...]
    missing: tuple[str, ...] = (
        "trusted reference generation and execution observation",
        "source-to-IR-to-object-to-executable invocation closure",
        "toolchain/package/runtime transitive closure",
        "actual hardware image/configuration and timer/input binding",
        "complete campaign member and final executable admission",
    )


def _mapping(path: Path) -> dict:
    # Duplicate fields must not acquire an interpretation different from a
    # protected producer's. Read only the exact frozen bytes.
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise StageGateError("duplicate numerical readback field")
            result[key] = value
        return result

    value = json.loads(path.read_bytes(), object_pairs_hook=pairs)
    if type(value) is not dict:
        raise StageGateError("numerical readback input must be a mapping")
    return value


def _private_paths(private: dict, sources: tuple[Path, ...]) -> tuple[Path, ...]:
    hosts = tuple(Path(row["destination"]) for row in private["marker"]["host_records"])
    for path in sources:
        if not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts:
            raise StageGateError("numerical readback paths must be absolute evaluator-owned inputs")
        if not any(path == host or host in path.parents for host in hosts):
            raise StageGateError("numerical readback input is not a protected host input")
    paths = BW.snapshot_input_paths(private["workspace"], private["bundle"], list(sources), repo=private["repo"])
    for path in paths:
        sha256_file(path)
        if path.stat().st_mode & 0o222:
            raise StageGateError("protected numerical readback input is writable")
    return tuple(paths)


def _signature(cb: dict) -> tuple[tuple[str, tuple[int, ...], str, int, bool], ...]:
    dtypes = declared_output_dtypes(cb)
    rows = []
    for name in cb["kernel_abi"]["outputs"]:
        dtype = dtypes.get(name, "")
        fmt = float_format_of(dtype)
        if fmt is not None:
            bits, signed = storage_bits(fmt), False
        else:
            bits = machine_bits(dtype)
            if not dtype.startswith(("i", "u")) or bits is None or not 1 <= bits <= 64:
                raise StageGateError("unsupported numerical readback output dtype")
            signed = not dtype.startswith("u")
        rows.append((name, tuple(cb["tensors"][name]["shape"]), fmt or dtype, bits, signed))
    return tuple(sorted(rows))


def _read_arm(files: tuple[Path, ...], service: BuildOnlyService, recipe: dict, pins: list[dict]):
    cb_path, receipt, kernel, harness, executable, console_path, _codec = files
    cb = _mapping(cb_path)
    # Parse duplicate keys before the existing owner validates its exact bytes.
    _mapping(receipt)
    RB.require_build_receipt(
        receipt,
        policy=RB.ReadbackPolicy(RB.FULL_VALUES_B64),
        cb=cb,
        target=service.target,
        recipe_record=recipe,
        source_pins=pins,
        object_path=kernel,
        harness_path=harness,
        elf_path=executable,
    )
    console = console_path.read_text(encoding="utf-8")
    raw, _metrics = parse_console(console)
    RB.require_full_value_roster(cb, console, raw)
    signature = _signature(cb)
    normalized = {}
    for name, _shape, dtype, bits, signed in signature:
        values = [value for row in raw[name] for value in row]
        floating = float_format_of(dtype) is not None
        # A signed float container is sign-extended on the wire; both legal
        # native representations denote the same stored bit pattern.
        lo = -(1 << (bits - 1)) if signed or floating else 0
        hi = (1 << bits) - 1 if floating or not signed else (1 << (bits - 1)) - 1
        if any(type(value) is not int or not lo <= value <= hi for value in values):
            raise StageGateError("output word exceeds its declared stored dtype")
        normalized[name] = tuple(value & ((1 << bits) - 1) for value in values)
    decoded = decode_float_readback(raw, declared_output_dtypes(cb))
    return signature, normalized, decoded


def admit_protected_numerical_readback(
    *,
    run_dir: Path,
    environment: Mapping,
    original_budget: Path,
    budget: QualityBudget,
    reference: ReadbackFiles,
    candidate: ReadbackFiles,
    reference_build: BuildOnlyService,
    candidate_build: BuildOnlyService,
) -> NumericalReadback:
    """Recompute exact/elementwise agreement using existing protected owners.

    Host-selected capabilities/provenance must come from the evaluator's own
    freeze/build lifecycle. A caller-authored V4 snapshot is not execution
    authentication. The original budget is a private frozen file containing
    QualityBudget.to_dict(); equality prevents tolerance/profile/reference
    substitution. Unsupported task metrics refuse instead of using a proxy.
    All values, both builds and private bytes are checked before and after use.
    """
    if type(budget) is not QualityBudget or budget.profile not in ("exact", "elementwise"):
        raise StageGateError("full readback requires a fixed exact or elementwise QualityBudget")
    if type(reference) is not ReadbackFiles or type(candidate) is not ReadbackFiles:
        raise StageGateError("readback inputs must be evaluator-owned ReadbackFiles")
    if type(reference_build) is not BuildOnlyService or type(candidate_build) is not BuildOnlyService:
        raise StageGateError("readback requires existing typed host build capabilities")
    provenance = copy.deepcopy(dict(environment))
    private = verify_private_input_snapshot(run_dir, provenance)
    sources = (original_budget, *reference.paths(), *candidate.paths())
    frozen = _private_paths(private, sources)
    identities = tuple((str(path), sha256_file(path)) for path in frozen)
    if RB.canonical_sha256(_mapping(frozen[0])) != RB.canonical_sha256(budget.to_dict()):
        raise StageGateError("fixed original quality budget differs from selected policy")
    records = []
    for service in (reference_build, candidate_build):
        service.verify(service.target)
        records.append(RB.selected_build_inputs(service.target, service.recipe, service))
    a = _read_arm(frozen[1:8], reference_build, *records[0])
    b = _read_arm(frozen[8:15], candidate_build, *records[1])
    if a[0] != b[0]:
        raise StageGateError("reference/candidate output shape or dtype roster differs")
    count = sum(len(values) for values in a[1].values())
    if count <= 0:
        raise StageGateError("numerical readback has no observed output elements")
    violations = 0
    for name, _shape, _dtype, _bits, _signed in a[0]:
        if budget.profile == "exact":
            violations += sum(x != y for x, y in zip(a[1][name], b[1][name], strict=True))
        else:
            params = dict(budget.parameters)
            x = [value for row in a[2][name] for value in row]
            y = [value for row in b[2][name] for value in row]
            if any(type(value) is int and abs(value) > 2**53 for value in (*x, *y)):
                raise StageGateError("elementwise float64 comparison cannot preserve this integer output")
            result = compare(y, x, **params)
            # Existing finite comparison semantics: NaN and infinity never
            # produce incidental elementwise passes, including inf == inf.
            violations += result["of"] - result["within"]
    after = verify_private_input_snapshot(run_dir, provenance)
    if _private_paths(after, sources) != frozen or any(sha256_file(Path(path)) != sha for path, sha in identities):
        raise StageGateError("protected readback bytes changed during numerical observation")
    for service, record in zip((reference_build, candidate_build), records, strict=True):
        if RB.selected_build_inputs(service.target, service.recipe, service) != record:
            raise StageGateError("selected readback build changed during numerical observation")
    metric = "bitwise_mismatch_count" if budget.profile == "exact" else "elementwise_violation_count"
    owner_paths = (private["bundle_path"], BW.bundle_snapshot_root(private["workspace"]) / "snapshot.json")
    selected_paths = {Path(row["path"]) for _recipe, pins in records for row in pins}
    selected_paths.update(Path(recipe["readback_codec"]["path"]) for recipe, _pins in records)
    evidence = tuple(
        sorted(set((*identities, *((str(path), sha256_file(path)) for path in (*owner_paths, *selected_paths)))))
    )
    observation = QualityObservation(
        ((metric, violations),),
        (
            "protected full numerical readback",
            private["marker"]["content_sha256"],
            RB.canonical_sha256(budget.to_dict()),
            *(sha for _path, sha in evidence),
        ),
        complete=True,
        parameters=budget.parameters,
    )
    return NumericalReadback(observation, count, tuple(row[0] for row in a[0]), evidence)
