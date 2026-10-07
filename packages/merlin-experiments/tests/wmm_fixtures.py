"""Harmless local fixtures for the ``whole_model_measured`` tests: a pinned file builder that writes a
fake "ELF" (a JSON program description) and a fake functional-model command that prints the whole-model
protocol for it.  Nothing here starts a simulator, a queue job or an agent."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from merlin_experiments.phase2.whole_model_measured import jobs as J
from merlin_experiments.phase2.whole_model_measured.identity import package_digest, program_digest

HEADER = "a" * 64

#: A sealed Phase 0 instruction policy for a synthetic target: what a config that declares
#: ``loop_descriptor`` must carry by value (``config.SEALED_POLICY``), resolved and non-vacuous.
SEALED_POLICY: dict[str, Any] = {
    "schema": "merlin.phase0.instruction_policy.v1",
    "prohibited_instruction_roles": ["loop_descriptor"],
    "status": "resolved",
    "prohibited_instructions": {"loop_descriptor": [{"selector": "8", "name": "LOOP_A"}]},
    "vacuous_roles": [],
    "taxonomy_status": "derived",
}


def sealed_policy(roles=("loop_descriptor",)) -> dict[str, Any]:
    """:data:`SEALED_POLICY` declaring ``roles``, each prohibiting one synthetic instruction."""
    roles = list(roles)
    return {
        **SEALED_POLICY,
        "prohibited_instruction_roles": roles,
        "prohibited_instructions": {r: [{"selector": str(8 + i), "name": f"LOOP_{i}"}] for i, r in enumerate(roles)},
    }


def write_phase0_manifest(root: Path, policy: dict[str, Any] | None = None) -> Path:
    """A sealed Phase 0 corpus ``MANIFEST.yaml`` carrying ``policy`` (default :data:`SEALED_POLICY`)."""
    import yaml

    path = Path(root) / "phase0_corpus" / "MANIFEST.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"instruction_policy": policy or SEALED_POLICY}), encoding="utf-8")
    return path


BUILDER_SOURCE = """
import hashlib, json
from pathlib import Path

def build(package_dir, *, target, out_dir, verify="on_target", prohibited_roles=(), **options):
    package_dir, out_dir = Path(package_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads((package_dir / "program.json").read_text())
    program = {"groups": spec["groups"], "argmax": spec.get("argmax", 3), "verify": verify}
    elf = out_dir / "program" / "model.elf"
    elf.parent.mkdir(parents=True, exist_ok=True)
    elf.write_text(json.dumps(program, sort_keys=True))
    (out_dir / "program" / "group_model_program.c").write_text("int main(){}\\n")
    record = out_dir / "build_record.json"
    record.write_text(json.dumps({"program": {"compiler": "/bin/true", "flags": []}, "linked_objects": []}))
    expectations = {
        "groups": {
            g: {"compare": "exact", "sum": v["want_sum"], "fnv1a": v["want_fnv"]} for g, v in spec["groups"].items()
        },
        "argmax": 3,
    }
    return {
        "elf": str(elf),
        "elf_sha256": hashlib.sha256(elf.read_bytes()).hexdigest(),
        "parameter_header_sha256": "HEADER",
        "expectations": expectations,
        "groups": [{"group": g, "op": "matmul", "on": v.get("on", "package")} for g, v in spec["groups"].items()],
        "notes": {"build_record": str(record)},
    }
""".replace("HEADER", HEADER)

SIM_SOURCE = """
import json, sys
program = json.loads(open(sys.argv[-1]).read())
print("MERLIN_INVOCATIONS warmup=1 measured=1")
total = 0
for g, v in sorted(program["groups"].items(), key=lambda kv: int(kv[0])):
    print(f"GM_GROUP {g} matmul {v['cycles']} sum={v['sum']} fnv1a={v['fnv']}")
    print(f"GM_WORDS {g} bytes=16 digest={v['sum']}")
    total += v["cycles"]
print(f"FM full model cycles: {total}")
print(f"GM_ARGMAX got={program['argmax']} want=3 agrees=1")
print("MERLIN_WINDOW end label=model")
"""


def write_builder(root: Path) -> tuple[str, str]:
    """``(spec, sha256)`` of a pinned file builder under ``root``."""
    path = Path(root) / "fake_builder.py"
    path.write_text(BUILDER_SOURCE)
    return f"{path}:build", hashlib.sha256(path.read_bytes()).hexdigest()


def write_sim(root: Path) -> list[str]:
    path = Path(root) / "fake_sim.py"
    path.write_text(SIM_SOURCE)
    return [sys.executable, str(path)]


def spike_machine(root: Path, target: str = "toy") -> dict[str, Any]:
    return {"kind": "spike", "target": target, "command": write_sim(root), "environment": {}, "identity_files": []}


def package(
    root: Path, name: str, *, groups: dict[str, dict[str, Any]] | None = None, argmax: int = 3, doc: str = ""
) -> Path:
    """A candidate package whose program.json the fake builder reads.  ``cycles``, ``sum`` and ``fnv``
    are what the program prints; ``want_*`` is the oracle."""
    groups = groups or {
        "1": {"cycles": 100, "sum": 7, "fnv": 9, "want_sum": 7, "want_fnv": 9},
        "2": {"cycles": 200, "sum": 5, "fnv": 6, "want_sum": 5, "want_fnv": 6},
    }
    pkg = Path(root) / name
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "program.json").write_text(json.dumps({"groups": groups, "argmax": argmax}, sort_keys=True))
    (pkg / "manifest.yaml").write_text("name: fake\n")
    if doc:
        (pkg / "docs").mkdir(exist_ok=True)
        (pkg / "docs" / "notes.md").write_text(doc)
    return pkg


def job_dir_for(store: Path, pkg: Path, *, machine: dict[str, Any], builder: tuple[str, str], **fields: Any) -> Path:
    """A job directory as the service writes one, without dispatching a worker."""
    import shutil

    digest = package_digest(pkg)
    job_dir = Path(store) / digest
    job_dir.mkdir(parents=True)
    shutil.copytree(pkg, job_dir / "package")
    job = {
        "schema": J.JOB_SCHEMA,
        "job_key": digest,
        "replicate": 0,
        "package_sha256": digest,
        "program_sha256": program_digest(pkg),
        "state": J.PENDING,
        "target": machine["target"],
        "builder": {"spec": builder[0], "sha256": builder[1]},
        "machine": machine,
        "build_options": {},
        "role": J.ROLE_CANDIDATE,
        "timeout_seconds": 60,
        **fields,
    }
    (job_dir / "job.json").write_text(json.dumps(job))
    return job_dir
