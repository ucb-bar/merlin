"""Phase 0 entry point of the chain-demand stage over pinned captured programs.

``derive_chains`` reads each selected program's frontend trace by an explicit sha256 pin, keeps only
its value-free structural graph (:func:`.chain_graph.read`), derives the plan of :mod:`.chain_demand`
and writes

* ``plan.json`` -- the digest-bound plan;
* ``chain-profile.yaml`` -- the correctness members as ordinary Phase 0 profile entries, which the
  generation reads as its ``chain_profile`` sidecar into the generated (correctness) cohort;
* ``sources/<case>/`` -- the performance members written by the ordinary Phase 0 writer (program,
  stimulus and integer-reference golden), the input the performance epoch admits into the public
  scalability suite.

The writer's binding is derived exactly as generation derives it: the descriptor's target, the
recipe's datapath with the selected software spec's numerical axes, and the RTL facts.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from . import chain_demand as D
from . import chain_graph as G
from .performance_corpus import parse_program


def read_program(model: str, program: str, path: Path, sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError(f"{model}/{program}: selected trace bytes changed")
    graph = G.read(json.loads(raw))
    del raw
    return graph


def writer_binding(*, descriptor: Path, recipe: Path, performance_template: Path, software_spec: Path,
                   facts: dict) -> tuple[Any, dict]:
    """The corpus binding and datapath an ordinary generation of this recipe selects."""
    from merlin.targetgen import corpus_spec as CS
    from merlin.targetgen.target_experiment import load_target_experiment

    from .profiles import load_profile

    te = load_target_experiment(descriptor)
    profile = load_profile(te.target, include_holdouts=False, recipe=recipe, performance_template=performance_template,
                           software_spec=software_spec, descriptor=descriptor)
    datapath = profile.get("datapath") or {}
    return CS.derive_binding(te, datapath, facts=facts), datapath


def derive_chains(programs: list[str], *, policy: Path, facts: Path, descriptor: Path, recipe: Path,
                  performance_template: Path, software_spec: Path, output: Path) -> dict[str, Any]:
    """Write ``plan.json``, ``chain-profile.yaml`` and ``sources/`` under a fresh ``output``."""
    from merlin.targetgen.target_experiment import load_target_experiment

    from .claim_boundary import held_out_models
    from .writer import _write_capsule

    if output.exists():
        raise ValueError("chain demand needs a fresh output directory")
    claims = set(held_out_models(load_target_experiment(descriptor)))
    selected = [parse_program(spec) for spec in programs]
    rows = [{"model": m, "program": p, "graph": read_program(m, p, path, sha), "trace_sha256": sha,
             "claim": m in claims} for m, p, path, sha in selected]
    facts_doc = json.loads(facts.read_text())
    plan = D.derive(rows, policy=yaml.safe_load(policy.read_text()), facts=facts_doc)
    plan["inputs"] = {
        name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in (("policy", policy), ("rtl_facts", facts), ("descriptor", descriptor), ("recipe", recipe),
                           ("performance_template", performance_template), ("software_spec", software_spec))
    }
    plan["inputs"]["programs"] = [{"model": m, "program": p, "trace_sha256": sha, "claim": m in claims}
                                  for m, p, _, sha in selected]
    binding, datapath = writer_binding(descriptor=descriptor, recipe=recipe, performance_template=performance_template,
                                       software_spec=software_spec, facts=facts_doc)
    narrow = binding.mlir_dtype(binding.requant_output_dtype or binding.operand_dtype)
    if not narrow.startswith("i") or not narrow[1:].isdigit():
        raise ValueError("chain cases need an integer readout format")
    output.mkdir(parents=True)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    entries = [D.member_entry(m, plan["sha256"], plan["policy"], narrow_bits=int(narrow[1:])) for m in plan["members"]]
    profile = {"capsules": [e for e in entries if e["cat"] != "sources"]}
    (output / "chain-profile.yaml").write_text(yaml.safe_dump(profile, sort_keys=False))
    semantics = datapath.get("numerical_semantics")
    written = []
    for item in (e for e in entries if e["cat"] == "sources"):
        item = copy.deepcopy(item)
        if semantics is not None:
            item["numerical_semantics"] = copy.deepcopy(semantics)
        path = _write_capsule(item, binding, output)
        if not path:
            raise ValueError(f"{item['name']}: the ordinary writer wrote no case")
        written.append(Path(path))
    return {"plan": str(output / "plan.json"), "plan_sha256": plan["sha256"],
            "chain_profile": str(output / "chain-profile.yaml"), "correctness_members": len(profile["capsules"]),
            "performance_sources": len(written), "deferred": len(plan["deferred"]),
            "members": {m["name"]: m["role"] for m in plan["members"]}}
