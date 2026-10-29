"""Protected complete operator prerequisites for single-capsule model requests.

No public counter, partial-tier union, historical compiler union or verdict cache
is authority. Reopen the original producer-owned products using the existing
cross-engine boundary checker. Missing/deleted/changed evidence grants no credit.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
import json
import os
from pathlib import Path

from . import model_gate_vouchers as vouchers, promotion, setup_timing, tier_boundary_join

_LIVE = ContextVar("rolling_model_gate_completion_owner", default=None)


def _pin(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("rolling gate evidence must be a regular original file")
    return sha256(path.read_bytes()).hexdigest()


def _qualified(path, capsule, scope, live_l2=None):
    """Full independently completed obligations only; split screens stay pending."""
    raw = Path(path).read_bytes()
    row = json.loads(raw)
    required = capsule.get("required_oracle_tiers", [])
    if (row.get("capsule") != capsule["name"] or row.get("kind") == "model"
            or row.get("status") != "pass" or row.get("failure")
            or not {"L2", "L3"} <= set(required)
            or row.get("numeric", {}).get("status") != "pass"):
        return None
    tiers = row.get("tiers") or {}
    from .joint_curriculum import _structural_prerequisites
    if not _structural_prerequisites(row):
        return None
    if any(tiers.get(t, {}).get("status") != "pass" for t in required if t not in {"L0", "L1"}):
        return None
    # This independently reopens the full typed outputs, input/numeric buffer,
    # source snapshot, actual producer invocation and each native ELF/object.
    right = tier_boundary_join.optional_observation(path, tier="L3")
    if right is None:
        return None
    if live_l2 is not None:
        _reopen_live_l2(live_l2)
        if (not tier_boundary_join.same_boundary(live_l2["common"], right["common"])
                or live_l2["tier_result"] != tiers["L2"]):
            raise ValueError("live L2 numerical/product boundary changed before full completion")
        joined = {"schema": "private_live_l2_complete_grade.v1", "common": right["common"],
                  "L2": live_l2, "L3": right}
    else:
        left = tier_boundary_join.optional_observation(path, tier="L2")
        if left is None:
            return None
        joined = tier_boundary_join.join(left, right)
    common = joined["common"]
    if (common["compiler_sha256"] != scope["compiler_sha256"]
            or common["target"] != scope["target"]
            or common["required_tiers"] != required):
        raise ValueError("rolling prerequisite source/target/required tiers differ")
    if Path(path).read_bytes() != raw:
        raise ValueError("rolling prerequisite changed during qualification")
    return joined


def _reopen_live_l2(record):
    """Only a previously live-verified ELF input may use its retained byte copy.

    Do not rewrite an invocation or relax its ordinary verifier. The original
    producer was fully verified before the next engine reused its build path.
    Every other input/dependency/output remains bound to the original path.
    """
    from merlin.common import invocation_record as I
    invocation = record["invocation"]
    if _pin(invocation["path"]) != invocation["sha256"]:
        raise ValueError("live L2 original invocation changed")
    document = json.loads(Path(invocation["path"]).read_bytes())
    if document != record["observed"]:
        raise ValueError("live L2 original producer record changed")
    replacement = record["retained_elf"]
    if _pin(replacement["path"]) != replacement["sha256"]:
        raise ValueError("live L2 retained executable changed")
    roles = [document["executable"], document["stdout"], document["stderr"],
             *document["inputs"], *document["outputs"], *document["dependencies"]]
    replaced = set(record["elf_input_paths"])
    if not replaced or not replaced <= {p["path"] for p in document["inputs"]}:
        raise ValueError("live L2 retained executable is not an original input")
    for pin in roles:
        if pin["path"] in replaced:
            if pin["sha256"] != replacement["sha256"]:
                raise ValueError("live L2 retained input differs from original producer")
        elif I._pin(Path(pin["path"])) != pin:
            raise ValueError("live L2 invocation dependency or output changed")


@setup_timing.measured("capture_tier")
def needs_original_execution(name, tier):
    """Whether :func:`capture_tier` needs this tier executed now (a carried verdict retains no custody)."""
    store = _LIVE.get()
    return store is not None and tier == "L2" and name in store.gate.denominator


def capture_tier(name, tier, result, *, generated, package_dir, shas):
    """Called by the original runner immediately AFTER its numerical comparison."""
    store = _LIVE.get()
    if store is None or tier != "L2" or name not in store.gate.denominator or result.status != "pass":
        return
    from merlin.common import invocation_record as I
    from merlin.targetgen import executed_program as E
    from merlin.common.tree_hash import hash_tree

    generated = Path(generated)
    try:
        store.source_check()
        if hash_tree(Path(package_dir))["sha256"] != store.scope["compiler_sha256"]:
            raise ValueError("live L2 source differs from selected compiler")
        tier_result = result.to_dict()
        records = [r for r in E.collect(generated) if r.get("status") == "retained"
                   and r.get("engine") == tier_result.get("engine")
                   and r.get("engine_sha256") == tier_result.get("sim_provenance", {}).get("sha256")]
        if len(records) != 1:
            raise ValueError("live L2 requires one retained original execution")
        execution = records[0]
        artifacts = execution["compiler_artifacts"]
        original = artifacts["original_execution_invocation"]
        if _pin(original["path"]) != original["sha256"]:
            raise ValueError("live L2 invocation differs from retained original")
        observed = I.verify(Path(original["path"]))
        elf = {"path": str(generated / execution["elf"]["path"]), "sha256": execution["elf"]["sha256"]}
        members = artifacts["members"]
        inputs = [p["path"] for p in observed["inputs"] if p["sha256"] == elf["sha256"]]
        if (observed.get("stage") != "execution" or not inputs
                or observed["stdout"]["sha256"] != members["console"]["sha256"]):
            raise ValueError("live L2 has no original executable/full-console producer")
        common = {"capsule": name, "target": store.scope["target"],
            "required_tiers": store.capsules[name]["required_oracle_tiers"],
            "compiler_sha256": store.scope["compiler_sha256"],
            "submission_digest": promotion.submission_tree_digest(Path(package_dir)),
            "toolchain_shas": shas,
            "source_ir_sha256": _pin(generated / "input.interface.mlir"),
            "raw_command_buffer_sha256": _pin(generated / "command_buffer.json"),
            "effective_command_buffer_sha256": _pin(generated / members["command_buffer"]["path"]),
            "lowered_ir_sha256": _pin(generated / members["lowered_mlir"]["path"]),
            "kernel_object_sha256": _pin(generated / members["kernel_object"]["path"])}
        record = {"schema": "private_live_numerically_checked_l2.v1", "common": common,
                  "tier_result": tier_result, "invocation": original, "observed": observed,
                  "retained_elf": elf, "elf_input_paths": inputs}
        _reopen_live_l2(record)
        store.live_l2[(name, str(generated.resolve()))] = record
    except (OSError, ValueError, KeyError, TypeError):
        # Optional scheduling evidence must never change the ordinary grade.
        # No record means no rolling credit; the model gate remains closed.
        store.live_l2.pop((name, str(generated.resolve())), None)


class CompletionStore:
    """Created only inside the ordinary original-profile worker execution scope."""
    def __init__(self, workspace, *, scope, capsules, gate, source_check):
        self.workspace, self.scope, self.capsules = Path(workspace), scope, capsules
        self.gate, self.source_check = gate, source_check
        self.key = sha256(vouchers._bytes(scope)).hexdigest()
        self.live_l2 = {}

    @contextmanager
    def observing(self):
        token = _LIVE.set(self)
        try:
            yield
        finally:
            _LIVE.reset(token)

    def _path(self, name):
        return vouchers._path(self.workspace, "operator-prerequisites-" + self.key, name)

    def load(self):
        self.source_check()
        for name in self.gate.denominator:
            path = self._path(name)
            if not path.exists():
                continue
            if path.is_symlink():
                raise ValueError("rolling gate private record is indirect")
            record = json.loads(path.read_bytes())
            if record.get("scope") != self.scope or record.get("capsule") != name:
                raise ValueError("rolling gate private record differs from selected identities")
            receipt = Path(record["receipt"])
            if _pin(receipt) != record["receipt_sha256"]:
                raise ValueError("rolling gate original completion changed")
            current = _qualified(receipt, self.capsules[name], self.scope, record.get("live_l2"))
            if current != record["qualification"]:
                raise ValueError("rolling gate original execution products changed")
            if current is not None:
                self.gate.qualified.add(name)
        self.source_check()

    def complete(self, result, path):
        name = result.get("capsule")
        if name not in self.gate.denominator:
            return
        self.source_check()
        raw = Path(path).read_bytes()
        if json.loads(raw) != json.loads(vouchers._bytes(result)):
            raise ValueError("rolling gate callback differs from durable completion")
        live_l2 = self.live_l2.get((name, str((Path(path).parent / "generated").resolve())))
        qualification = _qualified(path, self.capsules[name], self.scope, live_l2)
        self.source_check()
        record = {"schema": "private_operator_prerequisites.v1", "scope": self.scope,
                  "capsule": name, "receipt": str(Path(path).resolve()),
                  "receipt_sha256": sha256(raw).hexdigest(), "qualification": qualification, "live_l2": live_l2}
        destination = self._path(name)
        failure = result.get("failure") or {}
        pending_only = (result.get("status") == "incomplete"
            and result.get("numeric", {}).get("status") == "pass"
            and failure.get("plane") == "oracle_unavailable"
            and failure.get("category") == "NOT_RUN_IS_NOT_PASS"
            and not any(t.get("status") == "fail" for t in result.get("tiers", {}).values()))
        if qualification is None and pending_only and destination.is_file():
            # An intentionally partial screen supplies no new full-tier credit,
            # but cannot erase a complete qualification of identical bytes.
            # Actual failures still replace it. Admission reopens the old proof.
            previous = json.loads(destination.read_bytes())
            if previous.get("scope") == self.scope and previous.get("qualification") is not None:
                return
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            stream.write(vouchers._bytes(record)); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, destination)
        # A newly completed failure replaces earlier success in this exact scope.
        self.gate.qualified.discard(name)
        if qualification is not None:
            self.gate.qualified.add(name)


def broker_environment(workspace, capsules_root, submission):
    """Only the broker supplies these paths, after freezing its ordinary source."""
    from ..brokers.selfcheck import _submission_digest
    return {
        "MERLIN_SELFCHECK_MODEL_GATE_WS": str(Path(workspace).resolve()),
        "MERLIN_SELFCHECK_MODEL_GATE_ROOT": str(Path(capsules_root).resolve()),
        "MERLIN_SELFCHECK_MODEL_GATE_SHA256": _submission_digest(Path(submission)),
    }


def prepare(*, context, contract, source, runs, graded_root):
    """Bind the broker's private frozen source and full original policy cohort."""
    from . import snapshot_execution
    if snapshot_execution.selected():
        # Independent requests retain their numerical/ISA checks without an
        # unrelated cohort curriculum or fabricated prerequisite credit.
        return None, None
    from .compiler_profile_runtime import active_original_execution
    from .joint_physical_correctness import roster
    from merlin.targetgen.capsule_common import discover_capsules
    from merlin.targetgen.capsule_runner import _gate_counts
    from merlin.targetgen.model_gate_continuation import ModelGatePrerequisites
    from merlin.targetgen.target_experiment import load_target_experiment
    from ..brokers.selfcheck import _submission_digest
    from .loop_grading import _suite_for

    keys = ("MERLIN_SELFCHECK_PROMOTION_WS", "MERLIN_SELFCHECK_PROMOTION_ROOT", "MERLIN_SELFCHECK_PROMOTION_SHA256")
    values = tuple(os.environ.get(k) for k in keys)
    physical = tuple(os.environ.get(k.replace("PROMOTION", "MODEL_GATE")) for k in keys)
    if any(physical):
        values = physical
    if not any(values) or not roster():
        return None, None
    if not all(values):
        raise ValueError("rolling gate has incomplete trusted broker context")
    workspace, root, digest = values
    def source_check():
        if _submission_digest(Path(source)) != digest:
            raise ValueError("rolling gate frozen compiler changed")
    source_check()
    execution = active_original_execution()
    if execution is None:
        raise ValueError("rolling gate requires admitted compiler execution")
    owner = execution.owner
    te = load_target_experiment(context.descriptor)
    caps = discover_capsules((Path(root),), labels={"public", "dev"}, contract=contract)
    selected = roster()
    by_name = {c["name"]: c for c in caps if c["name"] in selected}
    if set(by_name) != selected:
        raise ValueError("rolling gate declared correctness cohort is incomplete")
    operators = [c for c in by_name.values() if c.get("kind") != "model"]
    eligible = [c for c in operators if _gate_counts({"capsule": c["name"]}, caps, te.target)]
    denominator = frozenset(c["name"] for c in (eligible or operators))
    declarations = {name: _pin(Path(c["__dir__"]) / "capsule.yaml") for name, c in by_name.items()}
    # The ordinary worker grades the frozen public projection, not the policy
    # declaration: privacy projection may remove capture metadata and add a tier
    # ceiling. Pin both under this same original-execution owner. A subset copy
    # must retain the exact public declaration bytes.
    public = {c["name"]: c for c in discover_capsules(
        (Path(graded_root),), labels={"public", "dev"}, contract=contract)}
    models = {n: c for n, c in by_name.items() if c.get("kind") == "model"}
    for name, policy in models.items():
        graded = public.get(name)
        if graded is None or any(graded.get(k) != policy.get(k)
                for k in ("kind", "gate", "required_oracle_tiers")):
            raise ValueError("rolling gate public model policy differs from original cohort")
    graded_models = {n: _pin(Path(public[n]["__dir__"]) / "capsule.yaml") for n in models}
    declaration_sources = {str(Path(c["__dir__"]) / "capsule.yaml"): declarations[n]
                           for n, c in by_name.items()}
    declaration_sources.update({str(Path(public[n]["__dir__"]) / "capsule.yaml"): digest
                                for n, digest in graded_models.items()})
    instruments = {t: promotion.cert_instrument(t, context=context)
                   for t in sorted({t for c in operators for t in c.get("required_oracle_tiers", [])})}
    if not instruments or not all(instruments.values()):
        raise ValueError("rolling gate requires exact declared-tier checker identities")
    from merlin.common.tree_hash import hash_tree
    scope = {"schema": "rolling_model_gate_scope.v1", "target": te.target,
             "execution_mode": getattr(execution, "execution_mode", "profile_strict_v1"),
             "compiler_sha256": hash_tree(Path(source))["sha256"],
             "profile_source_sha256": execution.source_sha256,
             "bundle_snapshot_sha256": owner.snapshot_digest,
             "profile_sha256": owner.profile["sha256"],
             "declarations": declarations, "graded_model_declarations": graded_models,
             "denominator": sorted(denominator),
             "instruments": instruments, **vouchers._context(context, "L3", contract)}
    if not scope["bundle_snapshot_sha256"] or not scope["profile_sha256"]:
        raise ValueError("rolling gate lacks exact selected corpus/library identity")
    gate = ModelGatePrerequisites(te.target,
        graded_models, denominator, set(), sha256(vouchers._bytes(scope)).hexdigest(), declaration_sources)
    def check_inputs():
        source_check()
        gate.verify_declarations()
    store = CompletionStore(workspace, scope=scope, capsules=by_name, gate=gate, source_check=check_inputs)
    # Operator-only requests issue records without reopening the entire cohort.
    # Models load once at admission; completion callbacks add current successes.
    def complete(row):
        path = Path(runs) / "runs" / _suite_for(te.target) / str(row.get("capsule")) / "capsule_result.json"
        store.complete(row, path)
    return store, complete
