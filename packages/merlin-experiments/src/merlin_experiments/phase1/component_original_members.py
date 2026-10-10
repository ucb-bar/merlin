"""Live original-reference/standard-IR inputs for ordinary candidate execution.

This private owner transports original tests, never candidate verdicts. Every
original call/cohort remains present, including unavailable formats and sources.
The source-preparation and independent runtime gates remain separate.
"""

from __future__ import annotations

import hashlib
import math
import weakref
from dataclasses import dataclass
from pathlib import Path

from merlin.common.jsonio import canonical_json
from merlin.common.strict_json import loads
from merlin.targetgen.contract.tensor_types import type_identity
from merlin.targetgen.original_reference_values import TypedReferenceTensor
from merlin_experiments.phase0 import component_execution_budget as E
from merlin_experiments.phase0 import original_reference_roster as R
from merlin_experiments.phase0 import original_reference_standard_ir as S
from merlin_experiments.phase0 import original_standard_ir_plan as P

SCHEMA = "merlin.original_candidate_members.v1"
MEMBER_SCHEMA = "merlin.original_candidate_member.v1"
_ISSUED = weakref.WeakKeyDictionary()
_IDENTITY = ("original_member_id", "graph_path", "node", "target", "cohort", "extent")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(path):
    path = Path(path).absolute()
    if path.resolve() != path or any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("original candidate owner requires a canonical unlinked path")
    return path


def _read(path, limit):
    path = R._plain(path)
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("original candidate input exceeds its explicit source/transfer budget")
    return raw


def _envelope(original, emitted, contract, name):
    metadata = contract.verify()
    abi = emitted["ordered_abi"]
    for role in ("inputs", "outputs"):
        if len(abi[role]) != len(metadata[role]):
            raise ValueError("original candidate member changed its complete ordered tensor ABI")
        for observed, selected in zip(abi[role], metadata[role], strict=True):
            if (
                observed["name"] != selected["name"]
                or canonical_json(observed["shape"]) != canonical_json(selected["shape"])
                or type_identity(observed["dtype"]) != type_identity(selected["dtype"])
            ):
                raise ValueError("original candidate member changed its complete ordered tensor ABI")
    return {
        "schema": MEMBER_SCHEMA,
        "name": name,
        "label": "hidden" if original["cohort"] == "withheld_transfer" else "public",
        "interface_mlir": "source.mlir",
        "original": emitted["original"],
        "call": original["call"],
        "ordered_abi": abi,
        "numeric_policy": contract.policy.record(),
        "reference_contract_sha256": contract.sha256,
    }


def _values(slot, value):
    """Accept complete logical values or the shared readback's exact row form."""
    if type(value) is not list:
        raise ValueError("original candidate output requires the complete logical element list")
    shape = slot["shape"]
    if any(type(row) is list for row in value):
        rows, cols = (math.prod(shape[:-1]), shape[-1]) if shape else (1, 1)
        if len(value) != rows or any(type(row) is not list or len(row) != cols for row in value):
            raise ValueError("original candidate output readback changed its complete declared row geometry")
        value = [item for row in value for item in row]
    if len(value) != math.prod(shape):
        raise ValueError("original candidate output lost an original element")
    return value


def _draft(standard_ir, destination):
    """Reopen complete live originals and budget all new transfer before copying."""
    if type(standard_ir) is not S.OriginalReferenceStandardIr:
        raise ValueError("original candidate members require the exact live standard IR owner")
    document = standard_ir.record()
    references = standard_ir.references
    original = P.required_members(references)
    selected = P.validate(loads(R._plain(standard_ir.selection).read_bytes()), references)
    policy = selected["execution_budget"]
    totals = dict(document["totals"]["execution"])
    reserved, rows = 0, []
    if len(document["members"]) != len(original["members"]):
        raise ValueError("original candidate membership lost an original source slot")
    for index, (reference, emitted) in enumerate(zip(original["members"], document["members"], strict=True)):
        identity = {key: reference[key] for key in _IDENTITY}
        if canonical_json(identity) != canonical_json(emitted["original"]) or emitted[
            "reference_member_sha256"
        ] != _sha(R._json(reference)):
            raise ValueError("original candidate membership changed a call/cohort/reference identity")
        name = "original_" + _sha(canonical_json(identity))
        row = {
            "source_slot": index,
            "name": name,
            "original": identity,
            "reference_member_sha256": emitted["reference_member_sha256"],
            "state": "unavailable",
            "required_unknowns": emitted["required_unknowns"],
            "reason": emitted.get("reason", reference.get("reason", "original source unavailable")),
        }
        if emitted["state"] == "source_reference_ir_checked" and reference["state"] == "reference_checked":
            contract = S._contract(reference, references)
            metadata = contract.verify()
            # Reuse the original owner\'s conservative complete typed-transfer
            # and reference costs; policy ceilings are never measured work.
            cost = P.cost(metadata)
            source = Path(emitted["products"]["source"]["path"])
            source_bytes = _read(source, selected["budget"]["max_source_bytes"])
            source_pin = {"path": str(_canonical(source)), "sha256": _sha(source_bytes)}
            if canonical_json(source_pin) != canonical_json(emitted["products"]["source"]):
                raise ValueError("original candidate source differs from its actual standard IR product")
            envelope = _envelope(reference, emitted, contract, name)
            size = len(source_bytes) + len(canonical_json(envelope)) + 1
            exceeded = E._exceeded(policy, cost, totals)
            if (
                len(original["members"]) > selected["budget"]["max_members"]
                or source.stat().st_size > selected["budget"]["max_source_bytes"]
                or reserved + size > selected["budget"]["max_total_source_bytes"]
            ):
                exceeded.append("whole original candidate source roster")
            if exceeded:
                row["reason"] = "original candidate transfer budget exceeded: " + ", ".join(exceeded)
            else:
                reserved += size
                for key in E._METRICS:
                    totals[key] += cost[key]
                root = destination / name
                row.update(
                    state="bound_input",
                    reason="exact original test inputs only; candidate execution remains required",
                    cost=cost,
                    capsule_root=str(root),
                    source=source_pin,
                    source_bytes=len(source_bytes),
                    envelope=envelope,
                    reference_products=reference["products"],
                    program_sha256=emitted["products"]["source"]["sha256"],
                    output_roster=[slot["name"] for slot in metadata["outputs"]],
                )
        rows.append(row)
    if len({row["name"] for row in rows}) != len(rows):
        raise ValueError("original candidate source membership is duplicated")
    return {
        "schema": SCHEMA,
        "standard_ir_sha256": standard_ir.sha256,
        "reference_roster_sha256": references.sha256,
        "reader_source": R._pin(Path(__file__)),
        "destination": str(destination),
        "original_product_roots": [
            value for value in (document.get("destination"), original.get("destination")) if value
        ],
        "members": rows,
        "totals": {"execution": totals, "copied_source_bytes": reserved},
        "scope": "original finite inputs only; source readiness and every candidate/hardware verdict remain separate",
    }


@dataclass(frozen=True, eq=False)
class OriginalCandidateMembers:
    standard_ir: S.OriginalReferenceStandardIr
    destination: Path
    receipt_json: bytes

    @property
    def sha256(self):
        return _sha(self.receipt_json)

    def verify(self):
        if _ISSUED.get(self) != self.sha256:
            raise ValueError("original candidate members require their actual live original input owner")
        destination = _canonical(self.destination)
        if not destination.is_dir():
            raise ValueError("original candidate owner is absent")
        if _read(destination / "members.json", len(self.receipt_json) + 1) != self.receipt_json + b"\n":
            raise ValueError("original candidate member record changed")
        actual = _draft(self.standard_ir, destination)
        if canonical_json(actual) != self.receipt_json:
            raise ValueError("original candidate members changed complete original source/reference membership")
        expected_files = {"members.json"}
        for row in actual["members"]:
            if row["state"] != "bound_input":
                continue
            source, declaration = Path(row["capsule_root"]) / "source.mlir", Path(row["capsule_root"]) / "capsule.yaml"
            for path in (source, declaration):
                R._plain(path)
                expected_files.add(path.relative_to(destination).as_posix())
            if (
                _sha(_read(source, row["source_bytes"])) != row["program_sha256"]
                or _read(declaration, len(canonical_json(row["envelope"])) + 1)
                != canonical_json(row["envelope"]) + b"\n"
            ):
                raise ValueError("original candidate source/envelope product changed")
        actual_files = set()
        for path in destination.rglob("*"):
            _canonical(path)
            if path.is_file():
                actual_files.add(path.relative_to(destination).as_posix())
            elif not path.is_dir():
                raise ValueError("original candidate owner contains a special file")
        if actual_files != expected_files:
            raise ValueError("original candidate owner changed its complete file roster")
        return actual

    def require_complete(self):
        record = self.verify()
        if any(row["state"] != "bound_input" for row in record["members"]):
            raise ValueError("original candidate member denominator contains unavailable source/reference slots")
        return record

    def member(self, capsule_root):
        record = self.verify()
        selected = [row for row in record["members"] if row.get("capsule_root") == str(Path(capsule_root))]
        if len(selected) != 1:
            raise ValueError("ordinary candidate execution lacks its exact original source member")
        return OriginalCandidateMember(self, selected[0]["source_slot"])


@dataclass(frozen=True)
class OriginalCandidateMember:
    owner: OriginalCandidateMembers
    source_slot: int

    def verify(self, capsule_root=None):
        if type(self.owner) is not OriginalCandidateMembers or type(self.source_slot) is not int:
            raise ValueError("original candidate execution needs the exact live original member")
        rows = self.owner.verify()["members"]
        if not 0 <= self.source_slot < len(rows):
            raise ValueError("original candidate source slot is outside the complete roster")
        row = rows[self.source_slot]
        if row["state"] != "bound_input" or capsule_root is not None and Path(row["capsule_root"]) != capsule_root:
            raise ValueError("original candidate source member is unavailable or substituted")
        return row

    def contract_inputs(self):
        self.verify()
        reference = self.owner.standard_ir.references.record_without_verification()["members"][self.source_slot]
        contract = S._contract(reference, self.owner.standard_ir.references)
        inputs = R._tensors(
            loads(
                _read(
                    reference["products"]["inputs"]["path"],
                    contract.budget.max_source_bytes + 2 * contract.budget.max_payload_bytes,
                )
            )
        )
        contract._inputs(inputs)
        return contract, inputs

    def compare_values(self, observed):
        contract, inputs = self.contract_inputs()
        expected = contract.verify()["outputs"]
        if type(observed) is not dict or set(observed) != {row["name"] for row in expected}:
            raise ValueError("original candidate output lost the complete original result roster")
        actual = tuple(
            TypedReferenceTensor.from_values(
                row["name"],
                row["dtype"],
                row["shape"],
                _values(row, observed[row["name"]]),
                byteorder=contract.output_byteorder,
            )
            for row in expected
        )
        report = contract.compare(inputs, actual)
        return {"status": "pass" if report["passed"] else "fail", "original_reference": report}


def prepare(*, standard_ir, destination):
    destination = Path(destination).absolute()
    _canonical(destination)
    if destination.exists():
        raise ValueError("original candidate member destination must be fresh")
    record = _draft(standard_ir, destination)
    protected = {Path(pin.path) for pin in standard_ir.source_pins} | {
        Path(value) for value in record["original_product_roots"]
    }
    for row in record["members"]:
        if row["state"] == "bound_input":
            protected.add(Path(row["source"]["path"]).parent)
            protected.update(Path(pin["path"]).parent for pin in row["reference_products"].values())
    if any(destination.is_relative_to(path) or path.is_relative_to(destination) for path in protected):
        raise ValueError("original candidate products overlap original source selection")
    destination.mkdir(parents=True, mode=0o700)
    for row in record["members"]:
        if row["state"] != "bound_input":
            continue
        root = Path(row["capsule_root"])
        root.mkdir(mode=0o700)
        (root / "source.mlir").write_bytes(_read(row["source"]["path"], row["source_bytes"]))
        (root / "capsule.yaml").write_bytes(canonical_json(row["envelope"]) + b"\n")
    raw = canonical_json(record)
    (destination / "members.json").write_bytes(raw + b"\n")
    owner = OriginalCandidateMembers(standard_ir, destination, raw)
    _ISSUED[owner] = owner.sha256
    owner.verify()
    return owner
