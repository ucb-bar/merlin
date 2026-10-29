"""Admission of the zero-initialized bf16/f16 f32-accumulation policy.

The installed admission and review records are trusted-host governance inputs, not
cryptographic authentication of a human. This module never creates them. Raw draft
checker reports remain abstracted: only a fresh checked pair PLUS exact installed
review/build admission may produce the separate gate claim. No saved receipt input.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from merlin.common.paths import build_dir, checkout_root

from .gate_errors import GateRejected, GateUnavailable, GateUnsupported
from .lean import bf16_f32acc as policy

PASS = "lower_bf16_matmul_f32acc"
ADMISSION = "merlin/contract/bf16_f32acc_admission.json"
REVIEW = "merlin/contract/reviews/bf16_f32acc.json"
DOMAIN = "zero-init-two-input-bf16-f16-ordered-f32acc-index64-v1"
EXACT = "MerlinVerify.Check.Bf16F32AccExact.check_sound"
SITES = "MerlinVerify.Check.Bf16F32AccSites.check_sound"
COVERAGE = "MerlinVerify.Check.Bf16F32AccSites.collectOps_coverage"
ROOTS = {
    **policy.CONTRACT_THEOREMS,
    policy.THEOREM: policy.BOUNDARY[1],
    f"{policy.BOUNDARY[1]}.qualify_sound": policy.BOUNDARY[1],
}
STD3 = {"propext", "Classical.choice", "Quot.sound"}
PIN_FILES = (
    "Spec/Bf16F32AccContract",
    "Rules/Bf16F32AccContract",
    "Rules/Bf16F32AccBody",
    "Rules/Bf16F32AccReference",
    "Rules/Bf16F32AccTensor",
    "Rules/Bf16F32AccSite",
    "Rules/Bf16F32AccNonvacuity",
    "Rules/W8A8TensorTrace",
    "Rules/W8A8TensorOperands",
    "Rules/W8A8Fill",
    "Check/Bf16F32Acc",
    "Check/Bf16F32AccExact",
    "Check/Bf16F32AccSites",
    "Check/ParametricAlpha",
    "Check/Alpha",
    "Check/ParametricModule",
    "Check/PolicyParametric",
    "Check/PolicyModule",
    "Check/PolicyExact",
)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(path: Path) -> dict:
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate key: {key}")
            out[key] = value
        return out

    result = json.loads(path.read_text(), object_pairs_hook=unique)
    if type(result) is not dict:
        raise ValueError(f"expected object: {path}")
    return result


def source_bindings(root: Path) -> dict[str, str]:
    """Source-only conservative project closure; never traverses .lake/build/SDK trees."""
    files = set((root / "merlin/lean/MerlinVerify").rglob("*.lean"))
    files.update((root / "src/merlin/verify").rglob("*.py"))
    files.update(
        root / p
        for p in (
            "merlin/lean/Main.lean",
            "merlin/lean/Audit.lean",
            "merlin/lean/lakefile.toml",
            "merlin/lean/lake-manifest.json",
            "merlin/lean/lean-toolchain",
            "merlin/lean/SPEC_LOCK",
            "merlin/lean/LEDGER.md",
            "merlin/contract/schemas/lean_ir.schema.json",
            "merlin/contract/schemas/lean_receipt.schema.json",
            "build_tools/scripts/prepare_bf16_review.py",
            "src/merlin/common/paths.py",
            "src/merlin/common/owned_process.py",
            "src/merlin/common/owned_children.py",
            "src/merlin/llvmlower/passes_xdsl.py",
            "src/merlin/runtime/backends/zephyr_model.py",
            "src/merlin/frontends/linalg_mlir.py",
            "src/merlin/xdsl_dialects/_common.py",
            "src/merlin/xdsl_dialects/fp8.py",
            "src/merlin/__init__.py",
            "src/merlin/common/__init__.py",
            "src/merlin/frontends/__init__.py",
            "src/merlin/xdsl_dialects/__init__.py",
            "src/merlin/runtime/__init__.py",
            "src/merlin/runtime/backends/__init__.py",
        )
    )
    return {str(p.relative_to(root)): _digest(p.read_bytes()) for p in sorted(files)}


def installed_root() -> Path:
    """Resolve governance and proof resources from this installed implementation.

    The ordinary repository/output helpers support caller-selected work roots.
    Such a root must never supply policy approval, proof sources or checker
    executables to this trusted gate. Matching aliases are harmless; a different
    owner refuses before opening governance records or tools.
    """
    root = checkout_root()
    if root is None or (root / "src/merlin/verify/bf16_policy_gate.py").resolve() != Path(__file__).resolve():
        raise GateUnavailable("bf16 policy requires its installed source owner")
    root = root.resolve()
    for name, expected in (
        ("MERLIN_REPO_ROOT", root),
        ("MERLIN_WORK_DIR", root),
        ("MERLIN_OUT_ROOT", root / "out"),
    ):
        selected = os.environ.get(name)
        if selected and Path(selected).resolve() != expected.resolve():
            raise GateRejected(f"bf16 trusted source owner cannot be overridden by {name}")
    return root


def require_admission(root: Path) -> dict:
    """Only the fixed, coordinator-installed record can select an approved policy.

    Absence refuses before any native/tool read. Attribution is an explicit trusted
    installation premise; hashes alone and caller booleans cannot grant approval.
    """
    owner = installed_root()
    if Path(root).resolve() != owner:
        raise GateRejected("bf16 admission must belong to the installed source owner")
    root = owner
    path = root / ADMISSION
    if not path.is_file():
        raise GateRejected("human_review_pending: no installed bf16 policy admission")
    try:
        admitted, review = _json(path), _json(root / REVIEW)
        if set(admitted) != {"schema", "domain", "review_sha256", "sources", "statements", "lean", "exporter"}:
            raise ValueError("unsupported admission fields")
        if admitted["schema"] != "merlin.bf16.policy.admission.v1" or admitted["domain"] != DOMAIN:
            raise ValueError("wrong policy admission schema/domain")
        if set(review) != {
            "schema",
            "decision",
            "domain",
            "reviewer",
            "independent_of_implementation",
            "record",
            "sources_sha256",
            "statements_sha256",
        }:
            raise ValueError("unsupported human review record fields")
        if (
            review["schema"] != "merlin.bf16.policy.review.v1"
            or review["decision"] != "approved"
            or review["domain"] != DOMAIN
            or review["independent_of_implementation"] is not True
            or not isinstance(review["reviewer"], str)
            or not review["reviewer"].strip()
            or not isinstance(review["record"], str)
            or not review["record"].strip()
        ):
            raise ValueError("independently attributable policy approval absent")
        if admitted["review_sha256"] != _digest((root / REVIEW).read_bytes()):
            raise ValueError("review record changed")
        sources = source_bindings(root)
        if admitted["sources"] != sources:
            raise ValueError("approved project sources changed")
        for key in ("sources", "statements"):
            bound = _digest(json.dumps(admitted[key], sort_keys=True, separators=(",", ":")).encode())
            if review[key + "_sha256"] != bound:
                raise ValueError(f"review does not bind {key}")
        lock = _json(root / "merlin/lean/SPEC_LOCK")
        for suffix in PIN_FILES:
            rel = "MerlinVerify/" + suffix + ".lean"
            if lock.get(rel) != sources.get("merlin/lean/" + rel) or rel not in lock:
                raise ValueError(f"missing or changed policy pin: {rel}")
        if set(admitted["statements"]) != set(ROOTS):
            raise ValueError("incomplete theorem statement roster")
        for name, statement in admitted["statements"].items():
            if (
                not isinstance(statement, str)
                or not statement
                or lock.get("statement:" + name) != _digest(statement.encode())
            ):
                raise ValueError(f"missing or changed statement pin: {name}")
        return admitted
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise GateRejected(f"bf16 policy admission refused: {exc}") from exc


def _attr(op: dict, name: str) -> dict:
    return op["props"].get(name, op["attrs"].get(name, {}))


def _children(op: dict):
    for region in op["regions"]:
        for block in region["blocks"]:
            yield from block["ops"]


def _half_contraction(op: dict) -> bool:
    # Conservative superset of the frozen Lean candidates: all half-result
    # named matmuls and multiply-add generic reductions, including malformed
    # operand segments/bodies. Ordinary sums/products are outside this policy.
    # Lean alone establishes actual eligibility.
    half = any(
        r["type"].get("kind") == "tensor"
        and r["type"].get("element") in ({"kind": "float", "format": "bf16"}, {"kind": "float", "format": "f16"})
        for r in op["results"]
    )
    if not half:
        return False
    if op["name"] in ("linalg.matmul", "linalg.batch_matmul"):
        return True
    iters = _attr(op, "iterator_types").get("elements", [])
    return (
        op["name"] == "linalg.generic"
        and any(a.get("value") == "reduction" for a in iters)
        and any(
            any(x["name"] == "arith.mulf" for x in block["ops"])
            and any(x["name"] == "arith.addf" for x in block["ops"])
            for region in op["regions"]
            for block in region["blocks"]
        )
    )


def source_sites(document: dict) -> list[dict]:
    """Independently enumerate full-source positions; unsupported nested sites refuse.

    A conservative superset of Lean candidates avoids treating an omitted half
    contraction as successful coverage. This is a host check, not a Lean proof.
    """
    if document.get("schema") != "merlin.lean.ir.v1":
        raise GateUnavailable("unsupported source export schema")
    sites = []

    def no_nested(op):
        if _half_contraction(op):
            raise GateUnsupported("half contraction outside the supported function-body site scope")
        for child in _children(op):
            no_nested(child)

    for fn in document["module"]["ops"]:
        if fn["name"] != "func.func":
            no_nested(fn)
            continue
        regions = fn["regions"]
        if not regions:  # declaration
            continue
        if len(regions) != 1 or len(regions[0]["blocks"]) != 1:
            raise GateUnsupported("only single-block defined functions are supported")
        name = _attr(fn, "sym_name").get("value")
        if not isinstance(name, str) or not name:
            raise GateUnavailable("malformed function name in export")
        for index, op in enumerate(regions[0]["blocks"][0]["ops"]):
            if _half_contraction(op):
                sites.append(
                    {
                        "function": name,
                        "source_operation": index,
                        "source_results": [r["id"] for r in op["results"]],
                        "qualification_passed": True,
                    }
                )
            for child in _children(op):
                no_nested(child)
    if not sites:
        raise GateUnsupported("no half contraction sites; not positive policy coverage")
    return sites


def validate_report(report: dict, source_export: dict) -> list[dict]:
    """SDK-free report consistency checks; NEVER grants authority to a saved report."""
    try:
        if (report["checker"], report["theorem"], report["claim_type"], report["policy"]) != (
            policy.CHECKER,
            policy.THEOREM,
            "policy",
            "bf16-f32acc-contract",
        ):
            raise GateUnavailable("different checker/root/policy")
        if type(report["index_bits"]) is not int or report["index_bits"] != 64:
            raise GateUnsupported("policy requires index64")
        if report["verdict"] != "abstracted" or report["policy_reviewed"] is not False:
            raise GateUnavailable("draft checker tried to grant approval")
        local = report["pass_boundary"]
        if (local["semantic_match_theorem"], local["site_qualification_theorem"]) != (EXACT, SITES):
            raise GateUnavailable("different complete-output/local roots")
        for key in ("checked", "semantic_match", "site_qualification_passed"):
            if type(local[key]) is not bool:
                raise GateUnavailable("non-Boolean checker result")
            if not local[key]:
                raise GateRejected(f"bf16 complete policy check refused: {key}")
        expected = source_sites(source_export)
        for count in (report["candidates"], local["expected_sites"], local["covered_sites"]):
            if type(count) is not int or count != len(expected):
                raise GateRejected("incomplete source-site denominator")
        # JSON equality alone treats True as 1; compare canonical encodings too.
        if json.dumps(local["site_obligations"], sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise GateRejected("source-site position/order/result coverage differs")
        licenses = report["sites"]
        if len(licenses) != len(expected) or any(
            set(x) != {"mul", "add"} or any(type(v) is not str or v not in ("none", "contract") for v in x.values())
            for x in licenses
        ):
            raise GateRejected("wrong site flags or license count")
        return expected
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise GateUnavailable(f"malformed bf16 checker report/export: {exc}") from exc


def _ref(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": _digest(data), "bytes": len(data)}


def _bound(ref: dict) -> None:
    if _ref(Path(ref["path"])) != ref:
        raise GateRejected("bound checker/input/report bytes changed")


def _tools(params: dict, admitted: dict | None = None) -> dict:
    """Use ordinary repository freshness and receipt identity, not a custom SDK provider."""
    from .lean import check
    from .pass_validation import CheckerTools

    sdk = Path(params["sdk_bin"]).resolve()
    # Preserve the venv entry point: resolving its symlink selects base-Python
    # package discovery rather than the configured compiler environment.
    python = Path(params["compiler_python"]).absolute()
    CheckerTools(sdk, python)
    if admitted is not None and check._lean_block() != admitted["lean"]:
        raise GateRejected("checker/spec identity differs from approved policy build")
    if admitted is not None and {**check._exporter_block(python), "round_trip": "identical"} != admitted["exporter"]:
        raise GateRejected("exporter identity differs from approved policy build")
    paths = {
        "lake": sdk / "lake",
        "lean": sdk / "lean",
        "compiler_python": python,
        "checker": check.lean_exe().resolve(),
        "audit": check.lean_exe().with_name("merlin-lean-audit").resolve(),
    }
    return {k: _ref(p) for k, p in paths.items()}


def project_objects(root: Path) -> list[dict]:
    """Current project import objects, separate from the compiled checker image."""
    if root != installed_root():
        raise GateRejected("project import inventory must belong to the installed source owner")
    base = build_dir() / "lean/build/lib/lean"
    paths = sorted(p for p in base.rglob("*.olean*") if p.is_file() and not p.name.endswith(".hash"))
    if not paths or not (base / "Main.olean").is_file():
        raise GateUnavailable("canonical project import objects unavailable")
    return [_ref(path) for path in paths]


def audit_candidate(tools: dict, work: Path, *, roots: dict | None = None, roots_mode: bool = False) -> dict:
    """Exact selected lake env/auditor; no implicit build or ~/.elan fallback."""
    from .lean import check

    selected = ROOTS if roots is None else roots
    env = {**os.environ, "PATH": str(Path(tools["lake"]["path"]).parent) + os.pathsep + os.environ.get("PATH", "")}
    artifacts = []

    def run(label, arguments):
        runner = subprocess.run
        kwargs = {"check": False}
        if roots_mode:
            from merlin.common import owned_process

            runner, kwargs = owned_process.run, {}
        proc = runner(
            [tools["lake"]["path"], "env", tools["audit"]["path"], *arguments],
            cwd=check.lean_root(),
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            **kwargs,
        )
        for suffix, output in (("stdout", proc.stdout), ("stderr", proc.stderr)):
            path = work / (label + "." + suffix)
            path.write_text(output)
            artifacts.append(_ref(path))
        if roots_mode:
            cleanup = getattr(proc, "owned_process_cleanup", None)
            if type(cleanup) is not dict or cleanup.get("complete") is not True:
                raise GateUnavailable("draft MACC auditor child cleanup is unknown")
            path = work / (label + ".cleanup.json")
            path.write_text(json.dumps(cleanup, sort_keys=True) + "\n")
            artifacts.append(_ref(path))
        if proc.returncode != 0:
            raise GateUnavailable(f"bf16 audit {label} failed: exit {proc.returncode}")
        return json.loads(proc.stdout.strip().splitlines()[-1])

    for ref in tools.values():
        _bound(ref)
    modules = sorted(set(selected.values()))
    if roots_mode:
        root = installed_root()
        objects = project_objects(root)
        result = run("roots", ["--roots", ",".join(selected), "Main", *modules])
        if (
            result.get("audit_scope") != "exact_roots_and_all_imported_project_declaration_safety"
            or result.get("roots") != list(selected)
            or type(result.get("theorems_audited")) is not int
            or result["theorems_audited"] != len(selected)
            or result.get("violations") != []
        ):
            raise GateRejected("draft MACC exact-root/declaration-safety audit refused")
        axioms, statements = result.get("root_axioms"), result.get("statements")
        if type(axioms) is not dict or set(axioms) != set(selected):
            raise GateUnavailable("incomplete draft MACC transitive axiom roster")
        if type(statements) is not dict or set(statements) != set(selected):
            raise GateUnavailable("incomplete draft MACC theorem statement roster")
        for name in selected:
            if type(statements[name]) is not str or not statements[name]:
                raise GateUnavailable("missing draft MACC theorem type")
            if type(axioms[name]) is not list or any(type(a) is not str for a in axioms[name]):
                raise GateUnavailable("malformed draft MACC transitive axioms")
            if not set(axioms[name]) <= STD3:
                raise GateRejected("draft MACC theorem is not standard-three-only")
        for ref in [*tools.values(), *artifacts, *objects]:
            _bound(ref)
        if project_objects(root) != objects:
            raise GateRejected("draft MACC imported project objects changed during audit")
        return {
            "audit_scope": result["audit_scope"],
            "axioms": axioms,
            "statements": statements,
            "closure": result,
            "artifacts": artifacts,
            "project_objects": objects,
        }
    closure = run("closure", ["Main", *modules])
    if (
        closure.get("violations") != []
        or type(closure.get("theorems_audited")) is not int
        or closure["theorems_audited"] <= 0
    ):
        raise GateRejected("checker dependency audit refused")
    statements = run("statements", ["--statements", ",".join(selected), *modules]).get("statements")
    if type(statements) is not dict or set(statements) != set(selected):
        raise GateUnavailable("incomplete live theorem statement roster")
    axioms = {}
    for i, (name, module) in enumerate(selected.items()):
        result = run(f"root-{i}", ["--axioms", name, module])
        ax = result.get("axioms")
        if (
            result.get("decl") != name
            or type(ax) is not list
            or any(type(a) is not str for a in ax)
            or not set(ax) <= STD3
        ):
            raise GateRejected("root audit is not standard-three-only")
        axioms[name] = ax
    for ref in tools.values():
        _bound(ref)
    return {"axioms": axioms, "statements": statements, "closure": closure}


def check_policy(job: dict[str, Any]) -> dict[str, Any]:
    """Worker entry: normal return means a fresh pair AND installed policy admission."""
    params = job.get("params")
    if type(params) is not dict or set(params) != {"sdk_bin", "compiler_python", "index_bits"}:
        raise GateUnsupported("explicit tools/index only; caller approval/receipt fields forbidden")
    if type(params["index_bits"]) is not int or params["index_bits"] != 64:
        raise GateUnsupported("bf16 policy gate supports index64 only")
    if any(type(params[k]) is not str or not params[k] for k in ("sdk_bin", "compiler_python")):
        raise GateUnsupported("explicit tool paths required")
    root = installed_root()
    admitted = require_admission(root)  # MUST precede tool/SDK access.
    try:
        before, after = _ref(Path(job["before"])), _ref(Path(job["after"]))
        if before["sha256"] == after["sha256"]:
            raise GateUnsupported("identity pair; no changed policy boundary")
        tools = _tools(params, admitted)
        from .gate_checks import _stage

        work = Path(job["directory"]) / "bf16-policy"
        work.mkdir(exist_ok=False)
        audited = audit_candidate(tools, work)
        if audited["statements"] != admitted["statements"]:
            raise GateRejected("live theorem statements differ from approval")
        axioms = audited["axioms"]

        # The receipt API's historical auditor prepends ~/.elan; use only the
        # already audited exact roots from the selected build instead.
        def auditor(name, module):
            if ROOTS.get(name) != module:
                raise GateRejected("unexpected audit root")
            return axioms[name]

        stage = _stage(
            work,
            "merlin.llvmlower.passes_xdsl." + PASS,
            Path(before["path"]).read_text(),
            Path(after["path"]).read_text(),
        )
        receipt = policy.check_stage(
            stage, work / "receipt", m2m_python=Path(params["compiler_python"]), index_bits=64, theorem_auditor=auditor
        )
        if receipt.get("verdict") == "unmeasured":
            raise GateUnavailable(receipt.get("reason", "checker did not complete"))
        if receipt.get("schema") != "merlin.lean.receipt.v1" or receipt.get("verdict") != "abstracted":
            raise GateUnavailable("unexpected policy receipt schema/status")
        if (receipt.get("checker"), receipt.get("theorem"), receipt.get("policy"), receipt.get("claim_type")) != (
            policy.CHECKER,
            policy.THEOREM,
            policy.POLICY,
            "policy",
        ):
            raise GateUnavailable("receipt obligation identity mismatch")
        if receipt.get("exporter", {}).get("round_trip") != "identical":
            raise GateUnsupported("authoritative export round trip refused")
        if receipt["lean"] != admitted["lean"] or receipt["exporter"] != admitted["exporter"]:
            raise GateRejected("checker/spec/exporter bindings changed")
        report = _json(Path(receipt["cert"]["path"]))
        exports = report["program_exports"]
        if len(exports) != 2 or receipt["exports"] != dict(zip(("before", "after"), exports, strict=True)):
            raise GateRejected("exports do not bind one complete boundary")
        for name, original in (("before", before), ("after", after)):
            bound = receipt[name]
            if any(bound[k] != original[k] for k in ("sha256", "bytes")):
                raise GateRejected("receipt differs from actual worker boundary")
            doc = _json(Path(receipt["exports"][name]["path"]))
            if any(doc["source"][k] != original[k] for k in ("sha256", "bytes")):
                raise GateRejected("export differs from actual source bytes")
        sites = validate_report(report, _json(Path(exports[0]["path"])))
        for ref in (before, after, receipt["before"], receipt["after"], receipt["cert"], *exports):
            _bound(ref)
        if require_admission(root) != admitted or _tools(params, admitted) != tools:
            raise GateRejected("admission/build changed during check")
        return {
            "schema": "merlin.bf16.policy.gate.v1",
            "status": "admitted",
            "claim_type": "policy",
            "domain": DOMAIN,
            "claim": (
                "complete output equals approved f32-accumulation reference under SSA/call laws; "
                "strict local initialized-input policy refinement"
            ),
            "original_source_equivalence": False,
            "whole_model_definedness": False,
            "before": before,
            "after": after,
            "index_bits": 64,
            "sites": sites,
            "theorems": list(ROOTS),
            "axioms": axioms,
            "admission": _ref(root / ADMISSION),
            "review": _ref(root / REVIEW),
            "receipt": str(work / "receipt/receipt.json"),
        }
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
        raise GateUnavailable(f"bf16 live policy check unavailable: {exc}") from exc
