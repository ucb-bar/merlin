"""Closed operator-selected public whole programs for ordinary Phase 1 exploration.

This selection does not change the original conformance roster or authorize RTL.
The original model owner retains source placement, host compilation and complete
output checking. Candidates request names, never paths or model substitutions.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from merlin.common.digest import sha256_file
from merlin.common.strict_json import loads
from merlin.targetgen.public_program_timing import is_public_program

KEY = "public_program_suite"
SCHEMA = "merlin.corpus.public_program_suite.v1"
POLICY = {"engine": "spike", "tiers": ["L0", "L1", "L2"], "purpose": "public_program_development"}


def _pin(value):
    from .preparation import ordinary_tree

    if type(value) is not dict or set(value) != {"path", "sha256"}:
        raise ValueError("public program selection needs an exact operator file pin")
    path = Path(value["path"])
    ordinary_tree(path)
    if (not path.is_absolute() or path != path.resolve() or not path.is_file()
            or sha256_file(path) != value["sha256"]):
        raise ValueError("public program operator input changed or is indirect")
    return path


def admit_placement(capsule, directory, *, target, evidence, source_context=None):
    """Freeze the ordinary written-source admission, retaining original numerics.

    The evidence must include the independently derived whole-program inventory
    of this capture. ``screen_written`` checks that original capture binding;
    this helper neither chooses a lane nor substitutes device support claims.
    """
    from merlin.targetgen.frozen_placement import read as read_placement

    from ..phase0.program_admission import accelerator_placement_refusal, screen_written

    if not is_public_program(capsule) or capsule.get("kind") != "model":
        raise ValueError("public program placement requires an original complete model source")
    kwargs = {"source_context": source_context} if source_context is not None else {}
    screen = screen_written(capsule, Path(directory), target=target, evidence=evidence, **kwargs)
    if not isinstance(screen, dict) or screen.get("status") != "admitted":
        reason = screen.get("reason") if isinstance(screen, dict) else "no written source program"
        raise ValueError(f"public program placement is not admitted: {reason}")
    refusal = accelerator_placement_refusal(screen)
    if refusal is not None:
        raise ValueError(f"public program placement is not admitted: {refusal}")
    admitted = dict(capsule, software_screen=screen)
    read_placement(admitted, directory)
    return admitted


def read(path, *, target):
    """Reopen the full selected tree, declarations and original source/reference bindings."""
    from merlin.targetgen.frozen_placement import read as read_placement

    from ..phase0 import operator_exclusion as exclusion
    from ..phase0.public_programs import verify_written
    from ..phase1.capability_inventory import _digest
    from ..runner import fingerprint
    from .preparation import ordinary_tree

    path = Path(path)
    ordinary_tree(path)
    if not path.is_absolute() or path != path.resolve() or path.stat().st_size > 1 << 20:
        raise ValueError("public program roster must be a bounded ordinary operator file")
    raw = path.read_bytes()
    document = loads(raw, max_bytes=1 << 20)
    if (type(document) is not dict
            or set(document) != {"schema", "target", "policy", "corpus", "members", "operator_exclusion"}
            or document["schema"] != SCHEMA or document["target"] != target or document["policy"] != POLICY
            or type(document["members"]) is not list or not 1 <= len(document["members"]) <= 9):
        raise ValueError("public programs require a finite explicit Spike-only selected roster")
    corpus = document["corpus"]
    if type(corpus) is not dict or set(corpus) != {"path", "sha256"}:
        raise ValueError("public program corpus requires an exact complete tree pin")
    root = Path(corpus["path"])
    ordinary_tree(root)
    if not root.is_absolute() or root != root.resolve() or fingerprint(root) != corpus["sha256"]:
        raise ValueError("public program source/input/reference tree changed")
    private_pin = document["operator_exclusion"]
    _pin(private_pin)
    protection = exclusion.select(private_pin["path"], private_pin["sha256"])
    exclusion.require_selection(protection, root)
    names, members = set(), {}
    for row in document["members"]:
        if type(row) is not dict or set(row) != {"member", "capsule_sha256"}:
            raise ValueError("public program member requires its original declaration pin")
        relative = Path(row["member"])
        if (relative.is_absolute() or len(relative.parts) != 2 or relative.as_posix() != row["member"]
                or any(part in {".", "..", "hidden"} for part in relative.parts)):
            raise ValueError("public program member path is not a selected public source")
        directory = root / relative
        cap = yaml.safe_load((directory / "capsule.yaml").read_bytes())
        if (not isinstance(cap, dict) or not is_public_program(cap)
                or cap.get("source_role") != "materialized_performance_capture"
                or cap.get("required_oracle_tiers") != POLICY["tiers"]
                or cap.get("oracle_tier_ceiling") is not None
                or cap.get("name") != directory.name or cap["name"] in names
                or _digest(cap) != row["capsule_sha256"]):
            raise ValueError("public program source kind, original tiers or declaration changed")
        # A selected path does not confer public identity or bypass private-source protection.
        verify_written(cap, directory)
        # Source/reference publication alone cannot execute through the original
        # owner. Require its ordinary byte-bound Phase 0 split before release.
        read_placement(cap, directory)
        exclusion.check_publication(protection, directory)
        names.add(cap["name"])
        members[cap["name"]] = directory
    if path.read_bytes() != raw or fingerprint(root) != corpus["sha256"]:
        raise ValueError("public program selection changed while checking")
    return document, members


def selected(experiment):
    """Only the original operator descriptor may select a program source roster."""
    binding = (experiment.workload_spec or {}).get(KEY)
    if binding is None:
        return None
    path = _pin(binding)
    document, members = read(path, target=experiment.target)
    return path, document, members


def stage(selection, destination, *, target):
    """Copy a verified selected tree into a fresh native source bundle without new authority."""
    from ..runner import fingerprint
    from .preparation import copy_input

    document, _ = read(Path(selection), target=target)
    destination = Path(destination).absolute()
    if destination.resolve().is_relative_to(Path(document["corpus"]["path"])):
        raise ValueError("public program staging cannot mutate the selected source tree")
    destination.mkdir(parents=True, exist_ok=False)
    copy_input(Path(document["corpus"]["path"]), destination / "corpus", private=True)
    # Frozen source custody must not depend on a historical private selector
    # path. Keep the same bytes and pin outside the public capsule/view trees.
    protection = document["operator_exclusion"]
    private_selector = destination / "private/operator_exclusion.json"
    copy_input(Path(protection["path"]), private_selector, private=True,
               expected_sha256=protection["sha256"])
    private_selector.chmod(0o600)
    document = dict(document, corpus={"path": str(destination / "corpus"),
                                      "sha256": fingerprint(destination / "corpus")},
                    operator_exclusion={"path": str(private_selector), "sha256": protection["sha256"]})
    path = destination / "selection.json"
    path.write_text(json.dumps(document, indent=2) + "\n")
    read(path, target=target)
    return {"path": str(path), "sha256": sha256_file(path)}


def public_view(selection, destination, *, target, verify=False):
    """Expose only declared source/loader/profile, never proof dependencies or output artifacts."""
    from ..runner import fingerprint
    from .preparation import copy_input

    _, members = read(Path(selection), target=target)
    destination = Path(destination)
    if not verify:
        destination.mkdir(parents=True, exist_ok=False)
    rows, allowed = [], {"suite.json"}
    for name, directory in sorted(members.items()):
        cap = yaml.safe_load((directory / "capsule.yaml").read_bytes())
        fields = {"source": cap["interface_mlir"]}
        # The original public-program verifier already binds these retained
        # local copies. Historical capture paths are provenance, never file grants.
        for pin, role, member in (("loader_sha256", "loader", "capsule.pytorch.py"),
                                  ("profile_sha256", "profile", "profile.json")):
            if cap["materialized_capture"].get(pin):
                if sha256_file(directory / member) != cap["materialized_capture"][pin]:
                    raise ValueError("public program local source copy changed")
                fields[role] = member
        public = {}
        for role, member in fields.items():
            relative = Path(member)
            if relative.name != member or relative.is_absolute():
                raise ValueError("public program source view requires capsule-local source members")
            output = f"{name}/{relative.name}"
            allowed.add(output)
            if not verify:
                copy_input(directory / member, destination / output)
            if sha256_file(directory / member) != sha256_file(destination / output):
                raise ValueError("public program source view differs from its original declared source")
            public[role] = {"path": output, "sha256": sha256_file(destination / output)}
        rows.append({"name": name, "kind": "model", "sources": public, "policy": POLICY})
    summary = {"schema": SCHEMA, "members": rows}
    index = destination / "suite.json"
    if not verify:
        index.write_text(json.dumps(summary, indent=2) + "\n")
    if json.loads(index.read_bytes()) != summary:
        raise ValueError("public program source catalogue changed")
    actual = {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()}
    if actual != allowed or any(p.is_symlink() for p in destination.rglob("*")):
        raise ValueError("public program source view includes undeclared or answer-bearing members")
    return {"path": str(destination), "sha256": fingerprint(destination)}
