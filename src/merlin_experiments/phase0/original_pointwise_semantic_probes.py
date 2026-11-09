"""Actual bounded same-original pointwise stress probes and strict replay.

Every original family survives source/budget/native failure. Supplementary
finite probes never replace an original guard/private slot or a domain proof.
"""

from __future__ import annotations

import hashlib
import math
import subprocess
import weakref
from dataclasses import dataclass
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.common.jsonio import canonical_json
from merlin.common.paths import module_source_path
from merlin.common.strict_json import loads
from merlin.targetgen import original_pointwise_stress as PS
from merlin.targetgen.original_operator_reference import OriginalReferenceBudget, prepare_original_reference
from merlin.targetgen.original_pointwise_reference import OriginalPointwiseReferencePolicy
from merlin.targetgen.original_pointwise_sources import pointwise_source

from . import component_execution_budget as E
from . import original_reference_plan as P
from . import original_reference_roster as R
from . import original_reference_standard_ir as S
from . import original_semantic_review_plan as Q
from .operator_schema_intake import _selection
from .rtl_intake import RtlIntakePin, _outside

SCHEMA = "merlin.original_pointwise_semantic_probes.v1"
_ISSUED = weakref.WeakKeyDictionary()
_CALL = ("original_member_id", "graph_path", "node", "target")


def _stimulus(contract, selected):
    return R._stimulus(contract, selected)


def _plans(standard, document):
    emitted = standard.record()
    original = standard.references.record_without_verification()
    selected = P.validate(loads(standard.references.selection.read_bytes()))
    schema = loads(standard.references.schema_intake.receipt_json)
    _, contracts, _ = R._drafts(
        original["defaults"], schema=schema, basis=standard.references.basis, selection=selected
    )
    from .original_semantic_review import _defaults, _palette

    defaults = _defaults(original)
    owners = {canonical_json(row["selector"]): row for row in document["owners"]}
    families, planned, seen, totals = [], {}, set(), dict.fromkeys(E._METRICS, 0)
    reserved = dict.fromkeys(E._METRICS, 0)
    for index, source in enumerate(original["members"]):
        contract = contracts.get(index)
        if (
            contract is None
            or source["state"] != "reference_checked"
            or emitted["members"][index]["state"] != "source_reference_ir_checked"
        ):
            continue
        selector = Q.selector(
            source["form"], defaults[(source["graph_path"], source["target"], source["call"]["schema"])]
        )
        owner = owners.get(canonical_json(selector))
        if owner is None:
            continue
        if canonical_json(owner["input_palettes"]) != canonical_json(_palette(selected, contract.policy)):
            raise ValueError("pointwise reviewed owner changed the original typed palette domain")
        measure = (
            Q.measure_pointwise_stress
            if type(contract.policy) is OriginalPointwiseReferencePolicy
            else Q.measure_stress
        )
        cost = measure(contract, selected)
        for metric in reserved:
            reserved[metric] += cost[metric]
    source_bytes = 0
    for index, source in enumerate(original["members"]):
        key = tuple(source[name] for name in _CALL)
        if key in seen:
            continue
        seen.add(key)
        family = {
            "original_call": list(key),
            "state": "unavailable",
            "reason": "no exact supported original pointwise owner",
            "owner": None,
            "members": [],
        }
        families.append(family)
        contract = contracts.get(index)
        if contract is None or type(contract.policy) is not OriginalPointwiseReferencePolicy:
            continue
        selector = Q.selector(
            source["form"], defaults[(source["graph_path"], source["target"], source["call"]["schema"])]
        )
        owner = owners.get(canonical_json(selector))
        if owner is None:
            continue
        family["owner"] = owner["id"]
        if canonical_json(owner["numerical_policy"]) != canonical_json(source["policy"]):
            raise ValueError("supplementary original stress cannot change its selected numeric policy")
        settings = owner["stress_probes"]
        try:
            rendered = pointwise_source(
                source["form"],
                extent=settings["extent"],
                max_tensor_elements=selected["reference_budget"]["max_tensor_elements"],
            )
            probe = prepare_original_reference(
                source["form"],
                rendered,
                extent=settings["extent"],
                policy=contract.policy,
                budget=OriginalReferenceBudget(**selected["reference_budget"]),
                output_byteorder=selected["byteorder"],
            )
            metadata = probe.verify()
            values = PS.samples(probe.policy, metadata["parameters"], settings["profile"])
        except ValueError as error:
            family["reason"] = "original pointwise stress source unavailable: " + str(error)
            continue
        count = math.prod(metadata["inputs"][0]["shape"])
        cases = (len(values) + count - 1) // count
        for case in range(cases):
            slot = sum(len(row["members"]) for row in families)
            row = {
                "slot": slot,
                "case": case,
                "state": "unavailable",
                "reason": None,
                "extent": settings["extent"],
                "profile": settings["profile"],
                "contract_sha256": probe.sha256,
                "source_sha256": hashlib.sha256(rendered.loader.encode()).hexdigest(),
                "metadata_sha256": hashlib.sha256(rendered.metadata_json.encode()).hexdigest(),
            }
            family["members"].append(row)
            palette = values[case * count :] + values[: case * count]
            chosen = {**selected, "input_palettes": [{"dtype": probe.policy.operand_dtypes[0], "values": palette}]}
            cost = Q.measure_pointwise_stress(probe, chosen)
            row["cost"] = cost
            size = len(rendered.loader.encode()) + len(rendered.metadata_json.encode())
            source_bytes += size
            exceeded = E._exceeded(
                document["execution_budget"], cost, {key: totals[key] + reserved[key] for key in totals}
            )
            if cases > settings["max_cases"]:
                exceeded.append("complete scalar partition cases")
            if len(original["members"]) > document["budget"]["max_members"]:
                exceeded.append("complete original member denominator")
            if exceeded:
                row["reason"] = "pointwise stress preallocation budget exceeded: " + ", ".join(exceeded)
            else:
                planned[slot] = (probe, chosen)
                for metric in totals:
                    totals[metric] += cost[metric]
    all_rows = [row for family in families for row in family["members"]]
    if (
        len(all_rows) > document["budget"]["max_probes"]
        or source_bytes > document["budget"]["max_total_probe_source_bytes"]
    ):
        planned.clear()
        totals = dict.fromkeys(E._METRICS, 0)
        for row in all_rows:
            row["reason"] = "complete pointwise probe/source denominator exceeds selected budget"
    return families, planned, totals, selected


def _paths(destination, slot):
    owner = destination / str(slot)
    return {
        key: owner / name
        for key, name in (
            ("source", "source.py"),
            ("metadata", "metadata.json"),
            ("inputs", "inputs.json"),
            ("reference", "reference.json"),
            ("actual", "actual.json"),
            ("comparison", "comparison.json"),
            ("stress", "stress.json"),
        )
    }


def _invocation(path, argv, inputs):
    raw = loads(R._plain(path).read_bytes())
    if raw["status"] == "completed":
        raw = I.require_environment(path, environment=R.D.ENVIRONMENT)
    elif raw.get("status") not in {"failed", "interrupted"}:
        raise ValueError("pointwise probes require a terminal actual native observation")
    for selected in [
        raw["executable"],
        *raw["inputs"],
        *raw["dependencies"],
        *raw.get("outputs", []),
        *(raw[key] for key in ("stdout", "stderr") if key in raw),
    ]:
        if R._pin(selected["path"]) != selected:
            raise ValueError("pointwise probe native inputs/tool/products changed")
    if (
        raw.get("schema") != I.SCHEMA
        or raw.get("kind") != "subprocess"
        or raw["argv"] != argv
        or raw["stage"] != "original_pointwise_stress_batch_native"
        or raw.get("environment") != I.environment_identity(R.D.ENVIRONMENT)
        or raw["inputs"] != I._pins(inputs)
    ):
        raise ValueError("pointwise probe invocation does not bind its complete original typed request")
    return raw


def _request(destination, planned):
    members = []
    inputs = []
    for slot in planned:
        paths = _paths(destination, slot)
        members.append({"slot": slot, **{key: str(paths[key]) for key in ("source", "metadata", "inputs", "actual")}})
        inputs.extend(paths[key] for key in ("source", "metadata", "inputs"))
    return {"schema": "merlin.original_pointwise_probe_request.v1", "members": members}, inputs


def _replay(standard, document, destination, invocation):
    families, planned, totals, _ = _plans(standard, document)
    request, inputs = _request(destination, planned)
    worker = module_source_path("merlin_experiments.phase0.original_pointwise_stress_observer")
    readers = [
        worker,
        module_source_path("merlin_experiments.phase0.original_pointwise_reference_observer"),
        module_source_path("merlin_experiments.phase0.original_reference_observer"),
    ]
    manifest, observation = destination / "request.json", destination / "observation.json"
    if canonical_json(loads(R._plain(manifest).read_bytes())) != canonical_json(request):
        raise ValueError("pointwise stress request changed complete original probe membership")
    python = _selection(Path(standard.references.schema_intake.record()["selection_path"]).read_bytes())["python"]
    argv = [python, "-I", str(worker), str(manifest), str(observation)]
    native = _invocation(Path(invocation), argv, [manifest, *readers, *inputs]) if planned else None
    if native and native["status"] == "completed":
        if observation.stat().st_size > document["budget"]["max_probe_observation_bytes"]:
            raise ValueError("native pointwise observation exceeds its selected byte budget")
        actual = loads(observation.read_bytes())
        if canonical_json(actual) != canonical_json(
            {"schema": "merlin.original_pointwise_probe_observation.v1", "slots": list(planned)}
        ):
            raise ValueError("pointwise probe observer lost exact complete output slots")
    for family in families:
        for row in family["members"]:
            slot = row["slot"]
            if slot not in planned:
                continue
            contract, chosen = planned[slot]
            paths = _paths(destination, slot)
            if (
                paths["source"].read_text() != contract.source.loader
                or paths["metadata"].read_text() != contract.source.metadata_json
            ):
                raise ValueError("pointwise stress source no longer matches its exact original form")
            tensors = _stimulus(contract, chosen)
            if canonical_json(loads(paths["inputs"].read_bytes())) != canonical_json(
                [R._tensor_record(t) for t in tensors]
            ):
                raise ValueError("pointwise stress changed actual finite original input bytes")
            expected = [R._tensor_record(t) for t in contract.evaluate(tensors)]
            if canonical_json(loads(paths["reference"].read_bytes())) != canonical_json(expected):
                raise ValueError("pointwise stress reference changed complete typed outputs")
            if native["status"] != "completed":
                row.update(state="unavailable", reason="actual original pointwise native process failed or interrupted")
            elif not paths["actual"].is_file():
                row.update(state="unavailable", reason="native pointwise output unavailable")
            else:
                if paths["actual"].stat().st_size > document["budget"]["max_probe_observation_bytes"]:
                    raise ValueError("native pointwise full output exceeds its selected byte budget")
                comparison = R._comparison(contract, tensors, paths["actual"])
                stress = PS.observe(contract, tensors)
                if canonical_json(loads(paths["comparison"].read_bytes())) != canonical_json(comparison):
                    raise ValueError("pointwise stress independent full comparison changed")
                if canonical_json(loads(paths["stress"].read_bytes())) != canonical_json(stress):
                    raise ValueError("pointwise stress traversal changed actual source/reference partitions")
                row.update(
                    state="probe_checked" if comparison["passed"] else "unavailable",
                    reason="complete bounded same-original native/reference stress comparison"
                    if comparison["passed"]
                    else "native/reference pointwise comparison refuted",
                    stress=stress,
                    realized_stress=PS.realized(stress),
                )
            row["products"] = {key: R._pin(path) for key, path in paths.items() if path.is_file()}
        if family["members"] and all(row["state"] == "probe_checked" for row in family["members"]):
            family.update(state="finite_probes_checked", reason="all selected original finite partition probes checked")
    return {
        "schema": SCHEMA,
        "reference_roster_sha256": standard.references.sha256,
        "standard_ir_roster_sha256": standard.sha256,
        "families": families,
        "logical_totals": totals,
        "invocation": R._pin(invocation) if invocation else None,
        "scope": (
            "Supplementary selected finite original stress cases; "
            "no original-slot substitution or domain/compiled/hardware admission."
        ),
    }


@dataclass(frozen=True, eq=False)
class OriginalPointwiseStressProbes:
    standard_ir: object
    review: Path
    destination: Path
    invocation: Path | None
    source_pins: tuple[RtlIntakePin, ...]
    receipt_json: bytes

    @property
    def sha256(self):
        return hashlib.sha256(self.receipt_json).hexdigest()

    def verify(self):
        if type(self.standard_ir) is not S.OriginalReferenceStandardIr or _ISSUED.get(self) != self.sha256:
            raise ValueError("pointwise stress probes require actual live original native preparation")
        for selected in self.source_pins:
            selected.verify()
        if R._plain(self.destination / "probes.json").read_bytes() != self.receipt_json + b"\n":
            raise ValueError("pointwise stress private product changed")
        actual = _replay(
            self.standard_ir, Q.validate(loads(self.review.read_bytes())), self.destination, self.invocation
        )
        if canonical_json(actual) != self.receipt_json:
            raise ValueError("pointwise stress probes changed exact source/cohort/native membership")

    def record(self):
        self.verify()
        return loads(self.receipt_json)


def prepare(*, standard_ir, review, document, forbidden, destination):
    if type(standard_ir) is not S.OriginalReferenceStandardIr:
        raise ValueError("pointwise stress probes require actual live original reference/standard sources")
    Q.validate(document)
    families, planned, _, _ = _plans(standard_ir, document)
    destination = Path(destination).absolute()
    _outside(destination, forbidden)
    if any(p.is_symlink() for p in (destination, *destination.parents)):
        raise ValueError("pointwise stress needs a fresh ordinary private destination")
    destination.mkdir(parents=True, mode=0o700, exist_ok=False)
    request, inputs = _request(destination, planned)
    for slot, (contract, chosen) in planned.items():
        paths = _paths(destination, slot)
        paths["source"].parent.mkdir(mode=0o700)
        paths["source"].write_text(contract.source.loader)
        paths["metadata"].write_text(contract.source.metadata_json)
        tensors = _stimulus(contract, chosen)
        R._write(paths["inputs"], [R._tensor_record(t) for t in tensors])
        R._write(paths["reference"], [R._tensor_record(t) for t in contract.evaluate(tensors)])
    manifest, observation = destination / "request.json", destination / "observation.json"
    R._write(manifest, request)
    readers = [
        module_source_path(name)
        for name in (
            __name__,
            PS.__name__,
            "merlin_experiments.phase0.original_pointwise_stress_observer",
            "merlin_experiments.phase0.original_pointwise_reference_observer",
            "merlin_experiments.phase0.original_reference_observer",
        )
    ]
    invocation = None
    if planned:
        python = _selection(Path(standard_ir.references.schema_intake.record()["selection_path"]).read_bytes())[
            "python"
        ]
        worker = readers[2]
        try:
            I.run(
                [python, "-I", str(worker), str(manifest), str(observation)],
                directory=destination,
                cwd=destination,
                stage="original_pointwise_stress_batch_native",
                inputs=(manifest, *readers[2:], *inputs),
                outputs=(observation, *[_paths(destination, slot)["actual"] for slot in planned]),
                env=R.D.ENVIRONMENT,
                capture_output=True,
                timeout=document["budget"]["probe_timeout_s"],
            )
        except subprocess.TimeoutExpired:
            pass
        invocation = next((destination / "invocations").glob("*/invocation.json"))
        native = loads(invocation.read_bytes())
        if native["status"] == "completed":
            for slot, (contract, chosen) in planned.items():
                paths = _paths(destination, slot)
                tensors = _stimulus(contract, chosen)
                R._write(paths["comparison"], R._comparison(contract, tensors, paths["actual"]))
                R._write(paths["stress"], PS.observe(contract, tensors))
    actual = _replay(standard_ir, document, destination, invocation)
    raw = canonical_json(actual)
    R._write(destination / "probes.json", actual)
    pins = tuple(
        RtlIntakePin("private-original-pointwise-stress", str(path), R._pin(path)["sha256"])
        for path in (Path(review), *readers)
    )
    result = OriginalPointwiseStressProbes(standard_ir, Path(review).absolute(), destination, invocation, pins, raw)
    _ISSUED[result] = result.sha256
    result.verify()
    return result
