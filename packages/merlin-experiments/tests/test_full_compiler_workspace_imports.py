"""A staged full-compiler tree works on its own: no public module reaches withheld code.

The full-compiler author receives the reviewed public source minus the withheld packages. A public
module that imports a withheld one does not fail loudly there: a guarded import degrades to an empty
answer (contraction extents were ``{}``, so output fusion refused every cut but ``bare``), and a
module-level one makes the module, and everything importing it, unusable. Both are invisible from
the operator's checkout, where every module is present. These tests stage the tree exactly as the
operator does and import it with ONLY that tree on the path and every withheld prefix refused.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from merlin_experiments.phase1 import full_compiler_workspace as F

from merlin.common.access import module_matches
from merlin.common.paths import repo_root

#: Public modules that cannot import in the staged tree, and why. Each needs code a reviewed access
#: policy declares grader/oracle-owned (``MODULE_ACCESS``), not a withheld package an analysis could
#: be moved out of. A new entry is a reviewed decision; a module leaving this set is progress.
DECLARED_ORACLE_DEPENDENT = {
    "merlin.llvmlower.staged_admission": "merlin.targetgen.contract.compile (declared oracle route)",
    "merlin.targetgen.contract.model_kernel_route": "merlin.targetgen.contract.compile (declared oracle route)",
    "merlin.targetgen.corpus_spec": "merlin.targetgen.capsule_dram (declared oracle)",
}

_CHILD = r"""
import importlib, importlib.abc, json, sys
root, refused, modules = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3])
current = [None]
attempts = {}

def refused_name(name):
    return any(name == p or name.startswith(p + ".") for p in refused)

class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if refused_name(name):
            attempts.setdefault(current[0], set()).add(name)
            raise ModuleNotFoundError("withheld from the public compiler: " + name, name=name)
        return None

sys.meta_path.insert(0, Refuse())
sys.path.insert(0, root)
for module in modules:
    current[0] = module
    error = None
    try:
        importlib.import_module(module)
    except BaseException as exc:  # noqa: BLE001 - every failure is reported, none ends the walk
        error = f"{type(exc).__name__}: {exc}"
    outside = sorted(
        name for name, loaded in list(sys.modules.items())
        if (name == "merlin" or name.startswith(("merlin.", "merlin_experiments")))
        and getattr(loaded, "__file__", None) and not str(loaded.__file__).startswith(root)
    )
    print(json.dumps({"module": module, "error": error, "refused": sorted(attempts.get(module, ())),
                      "outside": outside}), flush=True)
"""


def refused_prefixes() -> list[str]:
    """Everything ``F.forbidden`` refuses: the withheld prefixes and every declared grader/oracle module."""
    return sorted({*F.DENIED_PREFIXES, *(m for entry in F.MODULE_ACCESS for m in entry.modules)})


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    return F.prepare(repo_root(), tmp_path_factory.mktemp("full-compiler") / "public", review_id="import audit")


def _run_child(script: str, root: Path, *args: str, cwd: Path, timeout: int = 900, data=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "MERLIN_"))}
    env.update(data or {})
    return subprocess.run([sys.executable, "-I", "-B", "-c", script, str(root), *args],
                          capture_output=True, text=True, env=env, cwd=str(cwd), timeout=timeout)


def import_audit(selection) -> list[dict]:
    """Import every staged module in fresh interpreters that see only the staged tree."""
    root = selection.root
    modules = sorted(module for _path, _digest, module in selection.members if module)
    jobs = max(1, min(8, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 4))
    chunks = [modules[i::jobs] for i in range(jobs)]
    refused = json.dumps(refused_prefixes())

    def run(chunk):
        proc = _run_child(_CHILD, root, refused, json.dumps(chunk), cwd=root.parent)
        rows = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
        if len(rows) != len(chunk):
            raise AssertionError(f"import audit child stopped early: {proc.stderr[-2000:]}")
        return rows

    with ThreadPoolExecutor(jobs) as pool:
        return [row for rows in pool.map(run, chunks) for row in rows]


def test_every_staged_module_imports_without_reaching_withheld_code(staged):
    assert not [m for _p, _d, m in staged.members if m and F.forbidden(m)]
    rows = import_audit(staged)
    assert len(rows) > 100  # the whole public compiler, not a fixture
    withheld = {r["module"]: r["refused"] for r in rows
                if any(module_matches(n, p) for n in r["refused"] for p in F.DENIED_PREFIXES)}
    leaked = {r["module"]: r["outside"] for r in rows if r["outside"]}
    failed = {r["module"]: r["error"] for r in rows if r["error"]}
    assert not withheld, f"public modules reach withheld packages at import time: {withheld}"
    assert not leaked, f"modules resolved outside the staged tree: {leaked}"
    unexpected = {m: e for m, e in failed.items() if m not in DECLARED_ORACLE_DEPENDENT}
    assert not unexpected, f"public modules that do not import in the staged tree: {unexpected}"


def test_no_public_module_names_a_withheld_package_even_lazily(staged):
    """Static half of the same boundary: function-level and ``import_module`` literals included.

    The import walk only executes module bodies. A lazy import inside a function is what silently
    returned ``{}`` from the extent reader, so the source is read too.
    """
    import ast
    from importlib.util import resolve_name

    hits = {}
    for path, _digest, module in staged.members:
        if not module:
            continue
        package = module if path.endswith("__init__.py") else module.rpartition(".")[0]
        for node in ast.walk(ast.parse((staged.root / path).read_bytes())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                relative = "." * node.level + (node.module or "")
                base = resolve_name(relative, package) if node.level else node.module or ""
                names = [base, *(f"{base}.{a.name}" for a in node.names if a.name != "*")]
            elif (isinstance(node, ast.Call)
                  and getattr(node.func, "attr", getattr(node.func, "id", None)) == "import_module"
                  and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                names = [node.args[0].value]
            for name in names:
                if any(module_matches(name, p) for p in F.DENIED_PREFIXES):
                    hits.setdefault(module, set()).add(name)
    assert set(hits) <= KNOWN_OPERATOR_PATHS.keys(), {m: sorted(v) for m, v in hits.items()
                                                      if m not in KNOWN_OPERATOR_PATHS}
    for module, names in hits.items():
        assert names <= KNOWN_OPERATOR_PATHS[module], (module, sorted(names))


#: Public modules whose OPERATOR-only paths still name a withheld package lazily. Each fails loudly
#: (ModuleNotFoundError naming the package) or records why a field is empty; none answers ``{}``.
#: They are RVV host-lane flows that load mined schedule packages: the protected inputs themselves.
KNOWN_OPERATOR_PATHS = {
    "merlin.compile.host_lane": {"merlin.mining.registry", "merlin.mining.registry.DTYPE_STRATEGIES",
                                 "merlin.mining.registry.load_rvv_package"},
    "merlin.compile_cli": {"merlin.mining", "merlin.mining.k1", "merlin.mining.apply",
                           "merlin.mining.apply.blocking_risks", "merlin.mining.apply.shape_adapted_features",
                           "merlin.mining.registry", "merlin.mining.registry.load_rvv_package",
                           "merlin.compile.baremetal_model", "merlin.compile.baremetal_model.compile_saved_model"},
    "merlin.compile.owned_host_model": {"merlin.compile.baremetal_model",
                                        "merlin.compile.baremetal_model.compile_saved_model"},
    "merlin.runtime.source_recomputation_build": {"merlin.compile.baremetal_model"},
    "merlin.llvmlower.perop_blocks": {"merlin.mining.from_strategy",
                                      "merlin.mining.from_strategy._rvv_blocking_lowers",
                                      "merlin.mining.from_strategy._rvv_best_block"},
    "merlin.targetgen.target_experiment": {"merlin.mining.registry", "merlin.mining.registry.load_rvv_package"},
}


_FUSION = r"""
import importlib.abc, json, sys
root, refused, fixtures = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3])

class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if any(name == p or name.startswith(p + ".") for p in refused):
            raise ModuleNotFoundError("withheld from the public compiler: " + name, name=name)
        return None

sys.meta_path.insert(0, Refuse())
sys.path[:0] = [root, *fixtures]
from merlin.common import mlir_query as mq
from merlin.compile.analysis.contraction_shapes import contraction_extents
from merlin.targetgen import model_coverage as MC
from merlin.xdsl_dialects.lowering import compute_groups as CG
from merlin.xdsl_dialects.lowering import output_fusion as OF
import test_output_fusion as T

module = mq.parse(T._module())
extents = contraction_extents(module)
matmul = next(op for op in mq.walk(module) if mq.op_name(op) == "linalg.matmul")
regions = [(r.op, r.m, r.k, r.n, r.rank) for r in MC.regions_from_module(module) if r.op == "matmul"]
first = CG.form_groups(module, "synthetic", oracle=T._Oracle())
decided = OF.decisions(first, module, T._path(), OF.Values())
groups = CG.form_groups(module, "synthetic", oracle=T._Oracle(), fusion=decided)
(device,) = [g for g in groups if g.placement != CG.HOST]
print(json.dumps({"extent": list(extents.get(id(matmul), ())), "regions": regions,
                  "cut": device.fusion["cut"], "refusals": device.fusion["refusals"],
                  "sources": sorted({str(m.__file__).startswith(root) for n, m in list(sys.modules.items())
                                     if n.startswith("merlin") and getattr(m, "__file__", None)})}))
"""


def test_staged_tree_reads_extents_and_fuses_past_bare(staged):
    """The defect this guards: in the author tree the extent reader returned ``{}`` and output fusion
    refused every cut but ``bare`` with "reduction extent is not static"."""
    tests = repo_root() / "merlin" / "tests"
    fixtures = [str(tests / "fixtures"), str(tests / "ir")]
    # The staged tree carries code only. Read-only schema DATA (the quantized-format registry) is
    # named explicitly, as a sandbox names it; no withheld module is reachable through it.
    schemas = {"MERLIN_SCHEMAS_DIR": str(repo_root() / "merlin" / "schemas")}
    proc = _run_child(_FUSION, staged.root, json.dumps(refused_prefixes()), json.dumps(fixtures),
                      cwd=staged.root.parent, timeout=600, data=schemas)
    assert proc.returncode == 0, proc.stderr[-4000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["sources"] == [True]  # every merlin module came from the staged tree
    assert result["extent"] == [4, 8, 8, 3]
    assert result["regions"] == [["matmul", 4, 8, 8, 3]]
    assert result["cut"] == "requant", result["refusals"]


def test_bounded_rne_public_component_closes_without_target_runtime(tmp_path):
    """The custody leaf must freeze under the same exact import checker as SDK releases."""
    from merlin.targetgen.compiler_library import freeze_compiler_library

    entries = ("merlin.llvmlower.bounded_rne_custody", "merlin.llvmlower.host_transform_dispatch")
    component = F.prepare(repo_root(), tmp_path / "bounded-rne-public",
                          review_id="bounded RNE public dependency closure", modules=entries,
                          include_runtime_support=False)
    assert not component.diagnostics
    assert not any(m and F.forbidden(m) for _p, _d, m in component.members)
    assert not any(m and m.startswith("merlin.runtime.backends") for _p, _d, m in component.members)
    freeze_compiler_library(component.root, review_id="bounded RNE exact public import closure",
                            public_modules=entries, sources=tuple((p, m) for p, _d, m in component.members))
