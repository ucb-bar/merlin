"""Optional construction-plan admission, separate from compiler correctness.

Only selected private policy/inventory establish coverage. Author declarations
never establish implemented modules, worker activity, numerical passes or speed.
"""
from __future__ import annotations
from functools import lru_cache
from hashlib import sha256
import fcntl
import json
import os
from pathlib import Path, PurePosixPath

SCHEMA = "merlin.compiler_construction_plan.v1"
FILENAME = "CONSTRUCTION_PLAN.json"
EXAMPLE = "CONSTRUCTION_PLAN.example.json"
MAX_BYTES = 256 << 10


class PlanRequired(ValueError):
    pass


def policy(value):
    if value is None:
        return None
    if (not isinstance(value, dict) or value.get("required") is not True
            or set(value) != {"required", "bootstrap_capsules", "bootstrap_max_requests"}):
        raise ValueError("construction_plan requires required=true, bootstrap_capsules and bootstrap_max_requests")
    names = value["bootstrap_capsules"]
    if (not isinstance(names, list) or len(names) != 1 or any(type(n) is not str or not n or n == "all" for n in names)
            or type(value["bootstrap_max_requests"]) is not int or not 0 <= value["bootstrap_max_requests"] <= 3):
        raise ValueError("construction plan bootstrap requires one explicit tiny case and at most three requests")
    return value


def example(inventory, inventory_sha256):
    nodes = [{"id": f"capability_{i}", "owner": "compiler-author", "families": [family],
              "modules": [f"compiler/capability_{i}.py"], "depends_on": [],
              "contract": "Define typed inputs, outputs and shared compiler interface with dependent nodes."}
             for i, family in enumerate(sorted(inventory["families"]))]
    nodes.append({"id": "integration", "owner": "compiler-author", "families": [],
                  "modules": ["compiler/entrypoint.py"], "depends_on": [n["id"] for n in nodes],
                  "contract": "Integrate these modules into one compiler; original graders establish execution."})
    return {"schema": SCHEMA, "inventory_sha256": inventory_sha256,
            "nodes": nodes, "integration_node": "integration"}


def stage_example(workspace, run_dir, inventory, inventory_bytes):
    from .. import joint_control
    selected = joint_control.selected()
    option = policy((selected or {}).get("policy", {}).get("construction_plan"))
    if option is None:
        return
    bootstrap = option["bootstrap_capsules"][0]
    if bootstrap not in inventory["members"]:
        raise ValueError("construction-plan bootstrap is not in the selected public inventory")
    data = json.dumps(example(inventory, sha256(inventory_bytes).hexdigest()), indent=2) + "\n"
    for root in (Path(workspace), Path(run_dir)):
        (root / EXAMPLE).write_text(data)
        with (root / "TASK.md").open("a") as stream:
            stream.write("\nSelected construction-plan policy: after discovery, adapt `" + EXAMPLE
                + "` into `submission/" + FILENAME + "` before broad simulation requests. "
                "Keep inventory_sha256 and complete family coverage; choose owners, planned source paths, "
                "typed interface contracts, dependencies and an integration node reaching all nodes. "
                "Planned modules need not exist yet; this declaration is not implementation evidence. "
                f"Without a plan, at most {option['bootstrap_max_requests']} single-case requests for `{bootstrap}` "
                "are allowed across both brokers. Compile/diagnostic requests remain available.\n")


@lru_cache(maxsize=32)
def _validate(raw, inventory_raw):
    inventory = json.loads(inventory_raw)
    if not isinstance(inventory, dict) or inventory.get("schema") != "merlin.public_capability_inventory.v1":
        raise ValueError("selected public capability inventory schema differs")
    plan = json.loads(raw)
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA or plan.get("inventory_sha256") != sha256(inventory_raw).hexdigest():
        raise ValueError("construction plan must bind the exact selected public inventory")
    nodes = plan.get("nodes")
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 128:
        raise ValueError("construction plan needs 1..128 capability/integration nodes")
    by_id, coverage = {}, set()
    families = set(inventory["families"])
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("construction node must be an object")
        name = node.get("id")
        if (type(name) is not str or not name or len(name) > 80 or name in by_id
                or any(not (c.isalnum() or c in "_-.") for c in name)):
            raise ValueError("construction nodes require unique bounded IDs")
        for field, limit in (("owner", 128), ("contract", 2000)):
            value = node.get(field)
            if type(value) is not str or not value.strip() or len(value) > limit:
                raise ValueError("each construction node needs an owner and interface contract")
        modules = node.get("modules")
        if not isinstance(modules, list) or not 1 <= len(modules) <= 64:
            raise ValueError("each construction node needs planned module source paths")
        for module in modules:
            if (type(module) is not str or not module or len(module) > 240
                    or PurePosixPath(module).is_absolute() or ".." in PurePosixPath(module).parts
                    or str(PurePosixPath(module)) != module or module == "."
                    or any(not c.isprintable() for c in module)):
                raise ValueError("planned module paths must stay relative to the candidate source root")
        declared, dependencies = node.get("families"), node.get("depends_on")
        if (not isinstance(declared, list) or any(type(f) is not str for f in declared)
                or not set(declared) <= families or len(declared) != len(set(declared))
                or not isinstance(dependencies, list) or any(type(d) is not str for d in dependencies)
                or len(dependencies) != len(set(dependencies))):
            raise ValueError("construction families/dependencies must be explicit unique selected names")
        coverage.update(declared); by_id[name] = node
    if coverage != families:
        raise ValueError("construction plan omits public families: " + ", ".join(sorted(families - coverage)))
    if plan.get("integration_node") not in by_id:
        raise ValueError("construction plan needs an integration node")
    visited, active = set(), set()
    def visit(name):
        if name not in by_id or name in active:
            raise ValueError("construction dependencies are missing or cyclic")
        if name in visited:
            return
        active.add(name)
        for dependency in by_id[name]["depends_on"]: visit(dependency)
        active.remove(name); visited.add(name)
    visit(plan["integration_node"])
    if visited != set(by_id):
        raise ValueError("integration node must depend transitively on every construction node")
    return {"schema": "merlin.construction_plan_admission.v1", "mode": "declared_plan",
            "plan_sha256": sha256(raw).hexdigest(), "inventory_sha256": sha256(inventory_raw).hexdigest(),
            "declared_families": len(coverage), "required_families": len(families),
            "declared_nodes": len(nodes), "declared_owners": len({n["owner"] for n in nodes}),
            "observed_implementation": "not_assessed", "observed_workers": None,
            "observed_numerical_passes": None, "qualification_authority": False}


def _read(path, limit):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("construction plan/inventory must be a bounded ordinary file")
    with path.open("rb") as stream: raw = stream.read(limit + 1)
    if len(raw) > limit: raise ValueError("construction plan/inventory exceeds its byte bound")
    return raw


def readiness(submission):
    """Pure background predicate; no bootstrap spending or execution claim."""
    from .. import joint_control
    selected = joint_control.selected()
    if policy((selected or {}).get("policy", {}).get("construction_plan")) is None:
        return None
    try:
        run_dir = Path(os.environ[joint_control.ENV]).resolve().parent
        _validate(_read(Path(submission) / FILENAME, MAX_BYTES),
                  _read(run_dir / "PUBLIC_CAPABILITIES.json", 8 << 20))
    except (OSError, ValueError, TypeError, KeyError) as error:
        return f"construction plan not ready: {error}; adapt {EXAMPLE} into submission/{FILENAME}"
    return None


def require_ready(submission):
    reason = readiness(submission)
    if reason is not None:
        from .background_admission import SubmissionNotReady
        raise SubmissionNotReady(reason)


def admit(submission, request, *, request_id, channel, inspection=False):
    """Shared by both ordinary brokers, after their existing immutable snapshot."""
    from .. import joint_control
    selected = joint_control.selected()
    option = policy((selected or {}).get("policy", {}).get("construction_plan"))
    if option is None:
        return None
    # Cheap inspection and common tooling do not spend simulation capacity.
    if inspection:
        return {"mode": "inspection", "qualification_authority": False}
    run_dir = Path(os.environ[joint_control.ENV]).resolve().parent
    inventory_raw = _read(run_dir / "PUBLIC_CAPABILITIES.json", 8 << 20)
    plan_path = Path(submission) / FILENAME
    if plan_path.exists() or plan_path.is_symlink():
        try:
            report = dict(_validate(_read(plan_path, MAX_BYTES), inventory_raw))
        except (ValueError, TypeError, KeyError) as error:
            raise PlanRequired(f"{error}; adapt {EXAMPLE} into submission/{FILENAME}") from error
    else:
        names = str(request.get("capsules", "all")).split(",")
        bootstrap = option["bootstrap_capsules"]
        inventory = json.loads(inventory_raw)
        if len(names) != 1 or names != bootstrap or names[0] not in inventory["members"]:
            raise PlanRequired(f"broad execution requires submission/{FILENAME}; adapt {EXAMPLE}; "
                               f"one tiny bootstrap request may name {bootstrap[0]}")
        counter = run_dir / ".construction-bootstrap.json"
        with counter.open("a+") as stream:
            os.chmod(counter, 0o600); fcntl.flock(stream, fcntl.LOCK_EX)
            stream.seek(0); old = stream.read(); state = json.loads(old) if old else {}
            key = channel + ":" + str(request_id)
            if key not in state and len(state) >= option["bootstrap_max_requests"]:
                raise PlanRequired(f"bootstrap request budget exhausted; create submission/{FILENAME} from {EXAMPLE}")
            state[key] = {"capsule": names[0], "mode": "bootstrap"}
            stream.seek(0); stream.truncate(); json.dump(state, stream); stream.flush()
        report = {"mode": "bootstrap", "bootstrap_request": len(state),
                  "declared_families": 0, "required_families": len(inventory["families"]),
                  "observed_implementation": "not_assessed", "qualification_authority": False}
    # Protected append-only observations are descriptive, never gate authority.
    with (run_dir / "construction-plan-admissions.jsonl").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(dict(report, channel=channel, request_id=str(request_id))) + "\n")
    return report
