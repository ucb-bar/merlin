"""Explicit original-host preparation before compiler source/edit freezing.

Policy JSON selects no executable capability. The original live owner prepares
both treatments independently from the same immutable Phase 1 copy. Original
parent, derived comparison baseline and candidate seed keep distinct digests.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import MethodType

from merlin.benchharness import hash_tree
from merlin.targetgen.compiler_pass_selection import CompilerPassSelection
from merlin.targetgen.compiler_pass_selection import _tree as pass_tree

from . import campaign as PC
from .contracts import StageGateError

POLICY_SCHEMA = "merlin.compiler_pass_preparation.v1"
PAIR_SCHEMA = "merlin.prepared_compiler_pair.v1"


def _plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _is_sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


class CompilerPassPreparation:
    """Original host policy choices; not an admission or success constructor."""

    def __init__(self, mechanism, *, baseline_policy, candidate_policy):
        if type(mechanism) is not str or not mechanism or len(mechanism) > 128:
            raise StageGateError("compiler preparation requires one explicit bounded mechanism")
        if type(baseline_policy) is not dict or type(candidate_policy) is not dict:
            raise StageGateError("compiler preparation requires explicit policy mappings for both treatments")
        self._record = _plain(
            dict(
                schema=POLICY_SCHEMA,
                mechanism=mechanism,
                baseline_policy=baseline_policy,
                candidate_policy=candidate_policy,
            )
        )
        if len(json.dumps(self._record)) > 1 << 20:
            raise StageGateError("compiler preparation policy exceeds its bound")
        self.identity = _sha(self._record)

    def record(self):
        if _sha(self._record) != self.identity:
            raise StageGateError("original compiler preparation policy changed")
        return copy.deepcopy(self._record)


def validate_stage_selection(selection, preparation, library, diagnostic, search, edit_contract):
    """Preserve original ordinary-stage prerequisites before any source copying."""
    if selection is None and preparation is None:
        return
    if selection is None or type(preparation) is not CompilerPassPreparation or search is None or edit_contract is None:
        raise StageGateError(
            "production compiler passes require explicit preparation choices and queued edit authority"
        )
    if library is None or diagnostic or selection.binding.library != library.contract.sha256:
        raise StageGateError("prepared production treatments require the same explicit reviewed compiler library")


def read_preparation(path, expected_sha256):
    """Read bounded choices only after their original host-selected byte hash."""
    path = Path(path)
    if (
        not _is_sha(expected_sha256)
        or not path.is_absolute()
        or path.resolve() != path
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > 1 << 20
    ):
        raise StageGateError("compiler preparation requires an exact bounded ordinary policy leaf")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise StageGateError("compiler preparation policy hash differs from original selection")
    value = json.loads(raw)
    if type(value) is not dict or set(value) != {"schema", "mechanism", "baseline_policy", "candidate_policy"}:
        raise StageGateError("unknown compiler preparation policy fields")
    if value.pop("schema") != POLICY_SCHEMA:
        raise StageGateError("unknown compiler preparation policy schema")
    return CompilerPassPreparation(**value)


def _tree(path, *, readonly):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or path.is_symlink() or not path.is_dir():
        raise StageGateError("prepared compiler tree is missing, linked or noncanonical")
    members = (path, *path.rglob("*"))
    files = [member for member in members if member.is_file()]
    if len(files) > 8192 or sum(member.stat().st_size for member in files) > 32 << 20:
        raise StageGateError("prepared compiler tree exceeds its original source bound")
    if any(member.is_symlink() or not (member.is_file() or member.is_dir()) for member in members):
        raise StageGateError("prepared compiler tree contains an indirect or unsupported member")
    if readonly and any(member.stat().st_mode & 0o222 for member in members):
        raise StageGateError("prepared compiler or original parent is writable")
    if any({"build", "__pycache__", ".git"} & set(member.relative_to(path).parts) for member in members):
        raise StageGateError("prepared compiler retains unhashed ephemeral execution state")
    return str(hash_tree(path)["sha256"])


def _check_binding(selection, mechanism, package):
    capability = selection.resolve(mechanism)
    if capability is None:
        raise StageGateError("compiler preparation has no exact reviewed live capability")
    callback = getattr(capability.owner, "check_compiler_binding", None)
    if type(callback) is not MethodType or callback.__self__ is not capability.owner:
        raise StageGateError("compiler preparation requires the original bound compiler binding checker")
    callback(package)
    selection.revalidate()


def validate_pair(record, *, functional_sha256=None):
    """Pure custody format check; this never reconstructs an executable owner."""
    fields = {"schema", "selection", "preparation", "original", "baseline", "candidate_seed", "preparations", "sha256"}
    if type(record) is not dict or set(record) != fields or record["schema"] != PAIR_SCHEMA:
        raise StageGateError("unknown prepared compiler-pair record")
    value = {key: item for key, item in record.items() if key != "sha256"}
    if not _is_sha(record["sha256"]) or _sha(value) != record["sha256"]:
        raise StageGateError("prepared compiler-pair custody digest differs")
    policy = record["preparation"]
    if type(policy) is not dict or set(policy) != {"schema", "mechanism", "baseline_policy", "candidate_policy"}:
        raise StageGateError("prepared compiler-pair policy is incomplete")
    if policy.get("schema") != POLICY_SCHEMA:
        raise StageGateError("prepared compiler-pair policy has an unknown schema")
    expected = CompilerPassPreparation(**{k: v for k, v in policy.items() if k != "schema"}).record()
    if expected != policy or type(record["selection"]) is not dict:
        raise StageGateError("prepared compiler-pair lost its original selection")
    if record["selection"].get("mechanisms") != [policy["mechanism"]]:
        raise StageGateError(
            "compiler preparation requires one original composite capability, not chained preparations"
        )
    paths = []
    for role in ("original", "baseline", "candidate_seed"):
        arm = record[role]
        if type(arm) is not dict or set(arm) != {"path", "sha256"} or not _is_sha(arm["sha256"]):
            raise StageGateError("prepared compiler-pair lacks an exact " + role)
        path = Path(arm["path"])
        if not path.is_absolute() or str(path) != arm["path"] or ".." in path.parts:
            raise StageGateError("prepared compiler-pair has a noncanonical source path")
        paths.append(path)
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(paths) for b in paths[i + 1 :]):
        raise StageGateError("prepared treatments and original parent must be disjoint")
    if functional_sha256 is not None and record["original"]["sha256"] != functional_sha256:
        raise StageGateError("prepared compiler-pair has a different original Phase 1 parent")
    if type(record["preparations"]) is not dict or set(record["preparations"]) != {"baseline", "candidate_seed"}:
        raise StageGateError("prepared compiler-pair lacks both original preparation receipts")
    for receipt in record["preparations"].values():
        if (
            type(receipt) is not dict
            or receipt.get("schema") != "merlin.prepared_compiler_pass.v1"
            or receipt.get("selection") != record["selection"]
            or receipt.get("mechanism") != policy["mechanism"]
            or receipt.get("source") != record["original"]["path"]
        ):
            raise StageGateError("prepared compiler-pair contains a foreign original preparation receipt")
    return copy.deepcopy(record)


def verify_pair(record, selection, *, functional_sha256=None):
    """Reopen all immutable bytes under the explicitly selected original owner."""
    validate_pair(record, functional_sha256=functional_sha256)
    if type(selection) is not CompilerPassSelection or selection.record() != record["selection"]:
        raise StageGateError("prepared compiler-pair requires its exact live original pass selection")
    for role in ("original", "baseline", "candidate_seed"):
        arm = record[role]
        if _tree(arm["path"], readonly=True) != arm["sha256"]:
            raise StageGateError("prepared compiler-pair changed " + role + " bytes")
    for role in ("baseline", "candidate_seed"):
        receipt = record["preparations"][role]
        if receipt.get("source_sha256") != pass_tree(Path(record["original"]["path"])) or receipt.get(
            "prepared_sha256"
        ) != pass_tree(Path(record[role]["path"])):
            raise StageGateError("original preparation receipt has different source or treatment bytes")
        _check_binding(selection, record["preparation"]["mechanism"], Path(record[role]["path"]))
    return copy.deepcopy(record)


def rederive_pair(record, selection, preparation, *, replay_root, forbidden_roots=()):
    """Fresh original callback application under separately selected host choices.

    The candidate record cannot choose the policies, capability or replay root.
    Pure custody validation and package/source hashes alone grant no authority.
    """
    if type(preparation) is not CompilerPassPreparation or preparation.record() != record.get("preparation"):
        raise StageGateError("prepared handoff requires its independently selected original host policies")
    root = Path(replay_root)
    protected = [Path(record[role]["path"]) for role in ("original", "baseline", "candidate_seed")]
    protected.extend(Path(path) for path in forbidden_roots)
    if (
        not root.is_absolute()
        or root.resolve() != root
        or root.exists()
        or root.is_symlink()
        or any(root.is_relative_to(path) or path.is_relative_to(root) for path in protected)
    ):
        raise StageGateError("original preparation replay requires a fresh disjoint host-selected root")
    verify_pair(record, selection)
    actual = prepare_pair(
        selection,
        preparation,
        original=Path(record["original"]["path"]),
        functional_sha256=record["original"]["sha256"],
        root=root,
    )
    if any(actual[role]["sha256"] != record[role]["sha256"] for role in ("original", "baseline", "candidate_seed")):
        raise StageGateError("fresh original preparation differs from recorded compiler treatments")
    return actual


def prompt_projection(record):
    """Publish policy/byte identities without original private owner paths."""
    validate_pair(record)
    return dict(
        schema="merlin.compiler_preparation_prompt.v1",
        mechanism=record["preparation"]["mechanism"],
        original_sha256=record["original"]["sha256"],
        baseline_sha256=record["baseline"]["sha256"],
        candidate_seed_sha256=record["candidate_seed"]["sha256"],
        baseline_policy=record["preparation"]["baseline_policy"],
        candidate_policy=record["preparation"]["candidate_policy"],
    )


def validate_prompt_projection(value, *, original_sha256, baseline_sha256, candidate_seed_sha256):
    fields = {
        "schema",
        "mechanism",
        "original_sha256",
        "baseline_sha256",
        "candidate_seed_sha256",
        "baseline_policy",
        "candidate_policy",
    }
    if (
        type(value) is not dict
        or set(value) != fields
        or value["schema"] != "merlin.compiler_preparation_prompt.v1"
        or value["original_sha256"] != original_sha256
        or value["baseline_sha256"] != baseline_sha256
        or value["candidate_seed_sha256"] != candidate_seed_sha256
    ):
        raise StageGateError("prepared prompt relabels original, comparison baseline or candidate seed")
    CompilerPassPreparation(
        value["mechanism"], baseline_policy=value["baseline_policy"], candidate_policy=value["candidate_policy"]
    ).record()


def handoff_selection(handoff):
    """Recheck the original selected pass owner across paired worker threads."""
    record = getattr(handoff, "compiler_pass_preparation", None)
    selection = getattr(handoff, "compiler_pass_selection", None)
    choice = getattr(handoff, "compiler_pass_preparation_selection", None)
    if record is None:
        if selection is not None or choice is not None:
            raise StageGateError("legacy handoff cannot acquire compiler pass preparation")
        return None
    if type(choice) is not CompilerPassPreparation or choice.record() != record["preparation"]:
        raise StageGateError("prepared handoff lost its original host policy selection")
    verify_pair(record, selection)
    return selection


def prepare_pair(selection, preparation, *, original, functional_sha256, root):
    """Apply the selected composite once per arm, before inventory/edit freeze."""
    if type(selection) is not CompilerPassSelection or type(preparation) is not CompilerPassPreparation:
        raise StageGateError("production compiler passes require explicit live selection and preparation choices")
    policy = preparation.record()
    if selection.mechanisms != (policy["mechanism"],):
        raise StageGateError(
            "compiler preparation requires one original composite capability, not chained preparations"
        )
    original, root = Path(original), Path(root)
    if _tree(original, readonly=True) != functional_sha256:
        raise StageGateError("compiler preparation original copy is not the exact Phase 1 parent")
    if (
        root.exists()
        or root.is_symlink()
        or not root.is_absolute()
        or root.resolve() != root
        or root.is_relative_to(original)
        or original.is_relative_to(root)
    ):
        raise StageGateError("compiler preparation requires a fresh original host-owned directory")
    root.mkdir(parents=True)
    record = dict(
        schema=PAIR_SCHEMA,
        selection=selection.record(),
        preparation=policy,
        original=dict(path=str(original), sha256=functional_sha256),
        preparations={},
    )
    for role, key in (("baseline", "baseline_policy"), ("candidate_seed", "candidate_policy")):
        working = root / "preparation" / role
        working.parent.mkdir(exist_ok=True)
        record["preparations"][role] = selection.prepare_compiler(
            policy["mechanism"], original, working, policy=policy[key]
        )
        _check_binding(selection, policy["mechanism"], working)
        snapshot = root / role
        digest = PC.materialize_readonly_tree(working, snapshot)
        record[role] = dict(path=str(snapshot), sha256=digest)
    if selection.record() != record["selection"] or preparation.record() != record["preparation"]:
        raise StageGateError("compiler preparation selection or choices changed during application")
    record["sha256"] = _sha(record)
    verify_pair(record, selection, functional_sha256=functional_sha256)
    path = root / "preparation.json"
    path.write_text(json.dumps(record, sort_keys=True, indent=2))
    path.chmod(0o444)
    return record
