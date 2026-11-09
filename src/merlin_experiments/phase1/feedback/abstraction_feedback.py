"""Author-visible abstraction admission, seed-fallback identity and named-shape sweeps.

Three advisory views over evidence an ordinary check already produces. None changes emission,
grading, certification, promotion or any pass/fail status; every failure here degrades to an
explicit ``unavailable`` field.

* **Admission records.** Checked scheduling consumers in the shared library record each
  admit/refuse decision (``library_sources.admission_point``) as one bounded stderr line in the
  compiler process. The host already retains each compiler invocation's stderr beside the case's
  artifacts (``generated/invocations/*/stderr.bin``); :func:`case_admission` reads those lines back
  as ``{api, case, admitted, failed_premise, fallback_route, count}``. A refusal means the caller's
  own original route was emitted for that decision.
* **Seed fallback.** :func:`fallback_to_seed` compares the case's emitted kernel artifact with what
  the selected seed compiler emits for the same staged input (the byte-identity test of the
  operator-side frozen use audit). Identical bytes mean this case received no optimization relative
  to the seed. The seed emission is computed once per input and cached in the selection's private
  cache. Without an operator selection the field is ``None`` with its reason.
* **Named-shape sweep.** :func:`coverage_sweep` builds only the shapes an author names, from one
  selected public template case, through a caller-supplied resizer (the study's ``resize_dense``),
  and reports per shape the admission records, seed-fallback flag, transfer widths and fences of
  the emitted kernel, plus an optional functional Spike check. It never reads, names or compares
  against any other case.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "merlin.phase1.abstraction_feedback.v1"
SWEEP_SCHEMA = "merlin.phase1.coverage_sweep.v1"
SEED_RECEIPT = "merlin.phase1.seed_emission.v1"
SEED_ENV = "MERLIN_SEED_EMISSION_SELECTION"
ARTIFACT = "lowered.llvm.mlir"
COMPILER_STAGES = frozenset(
    (
        "parse",
        "lower_interface_to_target",
        "emit_command_buffer",
        "emit_analysis_bundle",
        "emit_target_bundle",
        "emit_target_artifact",
        "lower_target_to_llvm",
    )
)
MAX_INVOCATIONS = 64
MAX_STDERR_BYTES = 1 << 20
MAX_RECORDS = 64
MAX_SHAPES = 6
MAX_EXTENT = 1 << 16
QUALIFICATION = (
    "advisory author feedback: library-recorded decisions and byte identity of emitted artifacts; "
    "no numerical, RTL, timing or performance qualification"
)


def _scrub(text, limit=300):
    from merlin.common.path_scrub import scrub_host_paths

    return scrub_host_paths(str(text))[:limit] if text is not None else None


def _sha(data):
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- admission records
def case_admission(generated, *, case):
    """Admission records of the compiler invocations retained for one case."""
    from merlin.compiler.schedule.library_sources import parse_admission_stream

    root = Path(generated) / "invocations"
    if not root.is_dir():
        return {"status": "unavailable", "reason": "no retained compiler invocations", "records": []}
    calls = []
    for record in sorted(root.glob("*/invocation.json"))[: MAX_INVOCATIONS * 4]:
        try:
            document = json.loads(record.read_bytes()[:MAX_STDERR_BYTES])
        except (OSError, ValueError):
            continue
        if not isinstance(document, dict) or document.get("stage") not in COMPILER_STAGES:
            continue
        calls.append((document.get("finished_ns") or 0, document["stage"], record.parent / "stderr.bin"))
    merged, dropped = {}, 0
    for _, stage, stderr in sorted(calls)[:MAX_INVOCATIONS]:
        try:
            with stderr.open("rb") as stream:
                text = stream.read(MAX_STDERR_BYTES).decode("utf-8", errors="replace")
        except OSError:
            continue
        rows, lost = parse_admission_stream(text)
        dropped += lost
        for row in rows:
            key = (
                row["api"],
                row["admitted"],
                row["failed_premise"],
                row["fallback_route"],
                json.dumps(row.get("shape"), sort_keys=True),
            )
            if key in merged:
                merged[key]["count"] += row["count"]
                if stage not in merged[key]["stages"]:
                    merged[key]["stages"].append(stage)
            elif len(merged) < MAX_RECORDS:
                merged[key] = {
                    "api": row["api"],
                    "case": case,
                    "admitted": row["admitted"],
                    "failed_premise": _scrub(row["failed_premise"]),
                    "fallback_route": _scrub(row["fallback_route"]),
                    "shape": row.get("shape"),
                    "partial_refusals": (
                        [_scrub(v) for v in row["partial_refusals"]] if row.get("partial_refusals") else None
                    ),
                    "count": row["count"],
                    "stages": [stage],
                }
            else:
                dropped += 1
    records = list(merged.values())
    return {
        "status": "reported" if records else "none_reported",
        "records": records,
        "admitted_apis": sorted({r["api"] for r in records if r["admitted"]}),
        "refused_apis": sorted({r["api"] for r in records if not r["admitted"]}),
        **({"dropped": dropped} if dropped else {}),
        "meaning": (
            "each record is one checked library consumer's decision in this case's compile; a refusal "
            "emitted the caller's own original route for that decision"
        ),
    }


# --------------------------------------------------------------------------- seed selection
@dataclass(frozen=True)
class SeedSelection:
    """An operator-selected private seed compiler copy and its emission cache."""

    package: Path
    sha256: str
    cache: Path


_SELECTED = {"value": None, "env": None}
_LOCK = threading.Lock()
_PACKAGES = {}


def select_seed(package, *, cache, expected_sha256=None):
    """Verify a private seed compiler copy; optionally require its original tree digest."""
    from merlin.common.tree_hash import hash_tree

    package, cache = Path(package).absolute(), Path(cache).absolute()
    if not (package / "manifest.yaml").is_file():
        raise ValueError("seed emission requires a package directory with manifest.yaml")
    if any(p.is_symlink() for p in (package, *package.parents, cache, *cache.parents)):
        raise ValueError("seed emission paths must be direct")
    digest = hash_tree(package)["sha256"]
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("seed compiler copy differs from its selected digest")
    cache.mkdir(parents=True, exist_ok=True)
    return SeedSelection(package, digest, cache)


@contextlib.contextmanager
def seeded(selection):
    """Select ``selection`` for this process while the scope is open."""
    if selection is not None and type(selection) is not SeedSelection:
        raise ValueError("seed scope requires an explicit SeedSelection")
    with _LOCK:
        previous, _SELECTED["value"] = _SELECTED["value"], selection
    try:
        yield selection
    finally:
        with _LOCK:
            _SELECTED["value"] = previous


def seed_selection():
    """The scoped selection, else one transported by ``MERLIN_SEED_EMISSION_SELECTION``, else None."""
    with _LOCK:
        if _SELECTED["value"] is not None:
            return _SELECTED["value"]
        raw = os.environ.get(SEED_ENV)
        if not raw:
            return None
        if _SELECTED["env"] is not None and _SELECTED["env"][0] == raw:
            return _SELECTED["env"][1]
    view = json.loads(raw)
    if type(view) is not dict or set(view) != {"package", "sha256", "cache"}:
        raise ValueError("seed emission selection is malformed")
    selection = select_seed(view["package"], cache=view["cache"], expected_sha256=view["sha256"])
    with _LOCK:
        _SELECTED["env"] = (raw, selection)
    return selection


def _seed_package(selection, contract):
    from merlin.targetgen import package_runtime as P

    key = (str(selection.package), selection.sha256, str(contract))
    with _LOCK:
        package = _PACKAGES.get(key)
    if package is None:
        package = P.load_package(selection.package, contract=contract)
        P.integrity_scan(package)
        P.build_package(package)
        with _LOCK:
            _PACKAGES[key] = package
    return package


def seed_emission(selection, interface, *, contract, timeout, artifact_name=ARTIFACT):
    """The seed's emission digests for one exact input, computed once and cached."""
    from merlin.targetgen.capsule_common import lower_interface

    text = Path(interface).read_bytes()
    key = _sha(b"\0".join((selection.sha256.encode(), text, artifact_name.encode(), str(contract).encode())))
    entry = selection.cache / key[:32]
    receipt = entry / "receipt.json"
    if receipt.is_file():
        return json.loads(receipt.read_bytes())
    work = Path(tempfile.mkdtemp(prefix="seed-", dir=selection.cache))
    document = {
        "schema": SEED_RECEIPT,
        "seed_sha256": selection.sha256,
        "interface_sha256": _sha(text),
        "artifact_name": artifact_name,
    }
    try:
        staged = work / "seed.interface.mlir"
        staged.write_bytes(text)
        generated = work / "generated"
        try:
            lower_interface(
                _seed_package(selection, contract),
                staged,
                generated,
                contract=contract,
                timeout=timeout,
                artifact_name=artifact_name,
            )
        except Exception as error:  # noqa: BLE001 -- a seed refusal is evidence for this shape
            document.update(status="seed_not_emitted", reason=_scrub(f"{type(error).__name__}: {error}"))
        else:
            document.update(
                status="emitted",
                artifact_sha256=_sha((generated / artifact_name).read_bytes()),
                command_buffer_sha256=_sha((generated / "command_buffer.json").read_bytes()),
            )
        (work / "receipt.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        with contextlib.suppress(OSError):
            shutil.rmtree(generated)
        try:
            work.rename(entry)
        except OSError:
            if not receipt.is_file():
                raise
    finally:
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)
    return document


def fallback_to_seed(generated, *, contract, timeout, artifact_name=ARTIFACT, selection=None):
    """Whether this case's emitted kernel is byte-identical to the seed's emission of the same input."""
    generated = Path(generated)
    try:
        selection = selection or seed_selection()
    except Exception as error:  # noqa: BLE001
        return {"fallback_to_seed": None, "basis": "seed selection unavailable: " + _scrub(error)}
    if selection is None:
        return {"fallback_to_seed": None, "basis": "no seed emission is selected for this check"}
    interface, artifact = generated / "input.interface.mlir", generated / artifact_name
    if not interface.is_file() or not artifact.is_file():
        return {"fallback_to_seed": None, "basis": "this case emitted no kernel artifact to compare"}
    try:
        seed = seed_emission(selection, interface, contract=contract, timeout=timeout, artifact_name=artifact_name)
    except Exception as error:  # noqa: BLE001
        reason = _scrub(f"{type(error).__name__}: {error}")
        return {"fallback_to_seed": None, "basis": "seed emission unavailable: " + reason}
    if seed.get("status") != "emitted":
        return {
            "fallback_to_seed": None,
            "basis": "the seed compiler emits no kernel for this input",
            "seed_reason": seed.get("reason"),
        }
    kernel = _sha(artifact.read_bytes()) == seed["artifact_sha256"]
    buffer = generated / "command_buffer.json"
    return {
        "fallback_to_seed": kernel,
        "seed_identity": {
            "kernel_identical": kernel,
            "command_buffer_identical": (
                _sha(buffer.read_bytes()) == seed["command_buffer_sha256"] if buffer.is_file() else None
            ),
            "compared": artifact_name,
            "seed_sha256": selection.sha256[:16],
            "basis": "byte identity of this case's emitted kernel with the seed compiler's emission of the same input",
            "meaning": (
                "true: this case's kernel equals the seed's, so it carries no optimization for this shape; "
                "false: the emitted kernel differs (not by itself an improvement)"
            ),
        },
    }


def case_feedback(generated, *, case, contract, timeout, artifact_name=ARTIFACT):
    """Both per-case views, never raising."""
    out = {}
    try:
        out["abstraction_admission"] = case_admission(generated, case=case)
    except Exception as error:  # noqa: BLE001
        out["abstraction_admission"] = {"status": "unavailable", "reason": _scrub(error), "records": []}
    try:
        out.update(fallback_to_seed(generated, contract=contract, timeout=timeout, artifact_name=artifact_name))
    except Exception as error:  # noqa: BLE001
        out.update(fallback_to_seed=None, basis="seed comparison unavailable: " + _scrub(error))
    if "basis" in out:
        out["fallback_to_seed_basis"] = out.pop("basis")
    return out


def attach_cases(observed, runs, *, contract, timeout, deadline=None, artifact_name=ARTIFACT):
    """Add the per-case views to every observed row in place, within an optional monotonic deadline."""
    for name in sorted(observed):
        row = observed[name]
        if not isinstance(row, dict):
            continue
        if deadline is not None and time.monotonic() >= deadline:
            row["abstraction_admission"] = {"status": "unavailable", "reason": "request deadline", "records": []}
            row["fallback_to_seed"] = None
            continue
        remaining = timeout if deadline is None else max(1, min(timeout, int(deadline - time.monotonic())))
        row.update(
            case_feedback(
                Path(runs) / name / "generated",
                case=name,
                contract=contract,
                timeout=remaining,
                artifact_name=artifact_name,
            )
        )


# --------------------------------------------------------------------------- emitted census
def _width(instruction):
    decoded = instruction.get("decoded") if isinstance(instruction.get("decoded"), dict) else {}
    rows, cols = decoded.get("rows", decoded.get("rs2.num_rows")), decoded.get("cols", decoded.get("rs2.num_cols"))
    return f"{rows}x{cols}" if type(rows) is int and type(cols) is int else "UNKNOWN"


def census(artifact_text, *, target):
    """Fences and data-movement transfer widths of one emitted kernel (decoded sites, per-call executions).

    Classes and operand fields come from the target's selected decoder; the semantic family of each
    class (``movement``, ``synchronization``) comes from the shared family vocabulary. Unresolved
    widths or counts stay UNKNOWN/None.
    """
    from merlin.targetgen.rocc.decode import decode_text
    from merlin.targetgen.semantic_families import from_isa_class

    trace = decode_text(artifact_text, target=target)
    fences = {"static_sites": 0, "dynamic": 0}
    transfers, unknown = {}, 0

    def add(total, executions):
        return None if total is None or type(executions) is not int else total + executions

    for instruction in trace.get("instructions") or []:
        klass, executions = str(instruction.get("class")), instruction.get("executions")
        family = from_isa_class(klass)
        if klass == "UNKNOWN":
            unknown += 1
        if family == "synchronization":
            fences["static_sites"] += 1
            fences["dynamic"] = add(fences["dynamic"], executions)
        elif family == "movement":
            row = transfers.setdefault(klass, {"static_sites": 0, "dynamic": 0, "widths": {}})
            row["static_sites"] += 1
            row["dynamic"] = add(row["dynamic"], executions)
            label = _width(instruction)
            row["widths"][label] = row["widths"].get(label, 0) + (executions if type(executions) is int else 0)
    return {
        "fences": fences,
        "transfers": dict(sorted(transfers.items())),
        "unknown_instructions": unknown,
        "basis": "decoded emitted kernel; widths are rows x columns per command and counts are per-call executions "
        "of proved loop trip counts (None when a count is unproved); no timing",
    }


# --------------------------------------------------------------------------- named-shape sweep
def parse_shapes(text, *, max_shapes=MAX_SHAPES):
    """``m=24:k=2048,m=64`` -> ``[{"m": 24, "k": 2048}, {"m": 64}]``; author-named extents only."""
    if type(text) is not str or not text.strip() or len(text) > 2048:
        raise ValueError("coverage sweep needs comma-separated shapes such as m=24:k=2048,m=64")
    shapes = []
    for token in text.split(","):
        shape = {}
        for part in token.strip().split(":"):
            key, sep, value = part.partition("=")
            key = key.strip()
            if not sep or not key.isidentifier() or key in shape or not value.strip().isdigit():
                raise ValueError(f"coverage sweep shape {token.strip()!r} must be name=integer pairs joined by ':'")
            number = int(value)
            if not 0 < number <= MAX_EXTENT:
                raise ValueError(f"coverage sweep extent {key}={number} is outside 1..{MAX_EXTENT}")
            shape[key] = number
        if shape in shapes:
            raise ValueError("coverage sweep names the same shape twice")
        shapes.append(shape)
    if not 0 < len(shapes) <= max_shapes:
        raise ValueError(f"coverage sweep accepts 1 to {max_shapes} shapes per request")
    return shapes


def _template(public_cases, name, allowed):
    root = Path(public_cases).absolute()
    if name not in allowed:
        raise ValueError("coverage sweep template must be one selected public training case")
    directory = root / name
    if (
        Path(name).name != name
        or directory.is_symlink()
        or not directory.is_dir()
        or directory.resolve().parent != root.resolve()
    ):
        raise ValueError("coverage sweep template is not a direct selected public case")
    capsule = directory / "capsule.yaml"
    interface = directory / "capsule.interface.mlir"
    if not capsule.is_file() or not interface.is_file():
        raise ValueError("coverage sweep template lacks its public capsule files")
    return directory, capsule, interface


def _synthetic_capsule(template_yaml, cb, *, name, directory, interface_text):
    import yaml

    document = yaml.safe_load(template_yaml.read_text())
    for key in ("software_screen", "integer_partial_sum_bound", "performance_demand", "chain_demand"):
        document.pop(key, None)
    tensors = cb.get("tensors") or {}
    for row in document.get("inputs") or []:
        if row.get("name") in tensors:
            row["shape"] = list(tensors[row["name"]]["shape"])
    document.update(
        name=name,
        interface_mlir="capsule.interface.mlir",
        label="public",
        source_reference="author-named coverage sweep shape derived from a public template",
    )
    directory.mkdir(parents=True)
    (directory / "capsule.interface.mlir").write_text(interface_text)
    (directory / "capsule.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
    return directory


def _numerical(submission, capsule_root, *, runs_root, context, contract, timeout):
    from merlin.targetgen import capsule_grade as CG
    from merlin.targetgen.capsule_common import tier_status

    from .public_scalability import _readback, _sim_via
    from .selfcheck import _adapters

    adapters, engine = _adapters("spike", context.target, _sim_via(context), **_readback(context))
    if engine != "spike" or set(adapters) != {"L2"}:
        return {"status": "unavailable", "reason": "no exact Spike-only functional adapter"}
    from contextlib import redirect_stderr, redirect_stdout

    with (Path(runs_root) / "grader.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
        score = CG.grade(
            submission,
            capsules_root=capsule_root,
            runs_root=runs_root,
            target=context.target,
            labels={"public", "dev"},
            contract=contract,
            oracle_adapters=adapters,
            timeout=timeout,
            max_workers=1,
        )
    rows = score.get("per_capsule") or []
    if len(rows) != 1:
        return {"status": "unavailable", "reason": "functional check produced no single case row"}
    row = rows[0]
    failure = row.get("failure") or {}
    return {
        "status": row.get("status"),
        "tiers": {t: tier_status(v) for t, v in (row.get("tiers") or {}).items()},
        **(
            {"failure_plane": _scrub(failure.get("plane")), "failure_category": _scrub(failure.get("category"))}
            if failure
            else {}
        ),
        "engine": "spike",
        "qualification": "functional Spike check of an author-named shape; no RTL, timing or grade credit",
    }


def coverage_sweep(
    *,
    submission,
    public_cases,
    allowed_templates,
    template,
    shapes,
    resize,
    output,
    contract,
    context,
    timeout,
    numerical=False,
    artifact_name=ARTIFACT,
    max_shapes=MAX_SHAPES,
):
    """Compile author-named shapes of one public template's family; optional functional check per shape.

    ``resize(cb, shape) -> cb`` is the caller's existing case-resizing path. Only ``template`` (one of
    ``allowed_templates`` directly under ``public_cases``) is read; shapes are labelled by position.
    """
    from merlin.targetgen import package_runtime as P
    from merlin.targetgen.capsule_common import lower_interface
    from merlin.targetgen.contract.interface_emit import emit_interface_mlir, parse_interface_mlir

    if type(shapes) is not list or not 0 < len(shapes) <= max_shapes:
        raise ValueError(f"coverage sweep accepts 1 to {max_shapes} shapes per request")
    directory, capsule_yaml, interface = _template(public_cases, template, set(allowed_templates))
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + timeout
    base = parse_interface_mlir(interface.read_text())
    package = None
    rows = []
    for index, shape in enumerate(shapes):
        label = f"sweep-{index:02d}"
        row = {"label": label, "shape": dict(shape)}
        rows.append(row)
        remaining = int(deadline - time.monotonic())
        if remaining <= 5:
            row.update(status="not_run", reason="request deadline")
            continue
        try:
            cb = resize(copy.deepcopy(base), dict(shape))
            text = emit_interface_mlir(cb)
        except Exception as error:  # noqa: BLE001 -- an unsupported resize is the author's answer
            row.update(status="not_built", reason=_scrub(f"{type(error).__name__}: {error}"))
            continue
        case_dir = _synthetic_capsule(
            capsule_yaml, cb, name=label, directory=output / "cases" / label, interface_text=text
        )
        generated = output / "compiled" / label / "generated"
        try:
            if package is None:
                package = P.load_package(submission, contract=contract)
                P.integrity_scan(package)
                P.build_package(package)
            lower_interface(
                package,
                case_dir / "capsule.interface.mlir",
                generated,
                contract=contract,
                timeout=max(1, remaining),
                artifact_name=artifact_name,
            )
        except Exception as error:  # noqa: BLE001 -- a compile refusal is reported, never raised
            row.update(status="compile_failed", reason=_scrub(f"{type(error).__name__}: {error}"))
        else:
            row["status"] = "compiled"
        row.update(
            case_feedback(
                generated,
                case=label,
                contract=contract,
                timeout=max(1, int(deadline - time.monotonic())),
                artifact_name=artifact_name,
            )
        )
        admission = row.get("abstraction_admission") or {}
        row["refusals"] = [
            {k: r[k] for k in ("api", "failed_premise", "fallback_route", "count")}
            for r in admission.get("records", [])
            if not r["admitted"]
        ]
        row["admitted_apis"] = admission.get("admitted_apis", [])
        if row["status"] == "compiled":
            try:
                row["emitted"] = census((generated / artifact_name).read_text(), target=context.target)
            except Exception as error:  # noqa: BLE001
                row["emitted"] = {"status": "unavailable", "reason": _scrub(f"{type(error).__name__}: {error}")}
        if numerical and row["status"] == "compiled":
            left = int(deadline - time.monotonic())
            if left <= 5:
                row["numerical"] = {"status": "not_run", "reason": "request deadline"}
            else:
                single = output / "numerical" / label
                single.mkdir(parents=True)
                shutil.copytree(case_dir, single / "cases" / label)
                try:
                    row["numerical"] = _numerical(
                        submission,
                        single / "cases",
                        runs_root=single / "runs",
                        context=context,
                        contract=contract,
                        timeout=left,
                    )
                except Exception as error:  # noqa: BLE001
                    row["numerical"] = {"status": "unavailable", "reason": _scrub(f"{type(error).__name__}: {error}")}
        elif not numerical:
            row["numerical"] = {"status": "not_requested"}
    return {
        "schema": SWEEP_SCHEMA,
        "template": template,
        "shapes": rows,
        "limits": {"max_shapes": max_shapes, "max_extent": MAX_EXTENT, "timeout_s": timeout},
        "scope": (
            "only the shapes named in this request, built from the named public template through the "
            "study's case-resizing path; no other case is read or compared"
        ),
        "qualification": QUALIFICATION,
        "timing_qualified": False,
        "all_pass": None,
    }
