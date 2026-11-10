"""Dependency-free source discovery shared by repository gates during the layout cutover.

Scan canonical source once, retaining archived extension layouts. Historical debt remains
keyed by its original path: relocating a file neither forgives it nor grows a ratchet/allowlist.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
from pathlib import Path

SOURCE_SCAN_ROOTS = ("src", "tools/timing/src", "merlin/python/merlin", "packages")
EXCLUDED_PARTS = frozenset({"_data", "__pycache__", ".venv", "venv", "_qa_ws", "build", "dist"})


def module_alias(path: Path, root: Path) -> tuple[str, Path] | None:
    """Resolve a literal, otherwise inert ``sys.modules`` compatibility identity.

    This is static source correspondence, not an access grant or proof of use.
    Any extra executable statement makes the source an ordinary module again.
    """
    try:
        source = path.read_text(encoding="utf-8")
        if "modules" not in source or "__name__" not in source:
            return None
        tree = ast.parse(source)
    except (OSError, UnicodeDecodeError, SyntaxError):
        return None
    body = list(tree.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            body.pop(0)
    owner = binding = None
    imported_sys = assigned = False
    for node in body:
        if isinstance(node, ast.Import) and len(node.names) == 1:
            if node.names[0].name != "sys" or node.names[0].asname:
                return None
            imported_sys = True
        elif isinstance(node, ast.ImportFrom) and not node.level and len(node.names) == 1:
            item = node.names[0]
            if node.module == "__future__" and item.name == "annotations":
                continue
            if owner is not None or not item.asname or not node.module:
                return None
            owner, binding = f"{node.module}.{item.name}", item.asname
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            if not imported_sys:
                return None
            expected = ast.parse("sys.modules[__name__] = owner").body[0].targets[0]
            if ast.dump(node.targets[0]) != ast.dump(expected):
                return None
            if not isinstance(node.value, ast.Name) or node.value.id != binding:
                return None
            assigned = True
        elif isinstance(node, ast.If) and binding is not None:
            expected = ast.parse(f'if __name__ == "__main__":\n    raise SystemExit({binding}.main())').body[0]
            if ast.dump(node) != ast.dump(expected):
                return None
        else:
            return None
    if not imported_sys or not assigned or owner is None:
        return None
    parts = owner.split(".")
    candidates = set()
    for package in source_packages(root):
        if package.name == parts[0]:
            target = package.joinpath(*parts[1:])
            for candidate in (target.with_suffix(".py"), target / "__init__.py"):
                if candidate.is_file():
                    candidates.add(candidate.resolve())
    if len(candidates) != 1:
        return None
    target = candidates.pop()
    return (owner, target) if target != path.resolve() and target.is_relative_to(root.resolve()) else None


def policy_path(path: str) -> str:
    """Stable debt identity for the same merlin module, including shared-namespace distributions."""
    # Physical organization does not clear a source's existing review debt.
    for canonical, former in (
        ("src/merlin/evaluation/frameworks", "src/merlin/baselines"),
        ("src/merlin/development/backend/optimization", "src/merlin/experiments/phase2"),
        ("src/merlin/development/backend", "src/merlin/experiments/phase1"),
        ("src/merlin/development/testgen", "src/merlin/experiments/phase0"),
        ("tools/timing", "packages/merlin-timing"),
    ):
        if path == canonical or path.startswith(canonical + "/"):
            return policy_path(former + path[len(canonical) :])
    if path == "src/merlin/capture/pretranspose.py":
        return policy_path("src/merlin/baselines/pretranspose_cli.py")
    paper = "src/merlin/compare/paper_measurement_"
    if path.startswith(paper) and path[len(paper) :] in {
        "protocol.py",
        "study.py",
        "contract.py",
        "build.py",
        "process.py",
        "execution.py",
        "verification.py",
    }:
        return policy_path(paper + "controller.py")
    parts = path.split("/")
    if len(parts) >= 3 and parts[0] == "examples" and parts[1] in {"compilation", "compiler-generation", "targets"}:
        leaf = "compiler" if parts[2] == "compiler-api" else parts[2]
        tail = parts[3:]
        if tail[:2] == ["backend-generation", "optimization"]:
            tail = ["phase2", *tail[2:]]
        elif tail[:1] == ["backend-generation"]:
            tail = ["phase1", *tail[1:]]
        elif tail[:1] == ["test-generation"]:
            tail = ["phase0", *tail[1:]]
        return policy_path("/".join(("examples", leaf, *tail)))
    if path.startswith("experiments/inputs/"):
        return "experiments/reference-data/" + path[len("experiments/inputs/") :]
    # Relocated algorithms retain their existing debt identities; no new allowance.
    relocated = {
        "src/merlin/compiler/quantization/contraction/contract.py": "merlin/python/merlin/semantic_compiler/precision_contract.py",
        "src/merlin/compiler/quantization/contraction/intake.py": "merlin/python/merlin/semantic_compiler/precision_intake.py",
        "src/merlin/compiler/quantization/contraction/values.py": "merlin/python/merlin/semantic_compiler/precision_values.py",
        "src/merlin/compiler/quantization/contraction/constants.py": "merlin/python/merlin/semantic_compiler/precision_constants.py",
        "src/merlin/compiler/quantization/contraction/transform.py": "merlin/python/merlin/semantic_compiler/precision_transform.py",
        "src/merlin/compiler/quantization/contraction/codegen.py": "merlin/python/merlin/semantic_compiler/precision_codegen.py",
        "src/merlin/compiler/quantization/contraction/scalar.py": "merlin/python/merlin/semantic_compiler/contraction_precision.py",
        "src/merlin/compiler/quantization/contraction/batched.py": "merlin/python/merlin/semantic_compiler/batched_contraction.py",
        "src/merlin/compiler/lower/native/arguments.py": "merlin/python/merlin/llvmlower/pipeline.py",
        "src/merlin/compiler/lower/native/features.py": "merlin/python/merlin/llvmlower/pipeline.py",
        "src/merlin/compiler/lower/native/invoke.py": "merlin/python/merlin/llvmlower/pipeline.py",
        "src/merlin/compiler/lower/native/pipeline.py": "merlin/python/merlin/llvmlower/pipeline.py",
        "src/merlin/compiler/lower/native/policy.py": "merlin/python/merlin/llvmlower/pipeline.py",
        "src/merlin/compiler/lower/native/recipe.py": "merlin/python/merlin/llvmlower/lowering_recipe.py",
        "src/merlin/compiler/lower/native/reports.py": "merlin/python/merlin/llvmlower/pipeline.py",
        "src/merlin/compiler/lower/native/runners.py": "merlin/python/merlin/llvmlower/pipeline.py",
        "src/merlin/compiler/lower/native/stages.py": "merlin/python/merlin/llvmlower/pipeline.py",
        "src/merlin/compiler/lower/native/stock.py": "merlin/python/merlin/llvmlower/stock_cli.py",
        "src/merlin/compiler/lower/native/transport.py": "merlin/python/merlin/llvmlower/llvm_dialect_product.py",
        "src/merlin/compiler/lower/native/vector.py": "merlin/python/merlin/llvmlower/pipeline.py",
        "src/merlin/evaluation/capsule/adapters.py": "merlin/python/merlin/targetgen/capsule_runner.py",
        "src/merlin/evaluation/capsule/availability.py": "merlin/python/merlin/targetgen/capsule_runner.py",
        "src/merlin/evaluation/capsule/bundle.py": "merlin/python/merlin/targetgen/capsule_bundle.py",
        "src/merlin/evaluation/capsule/bundle_worker.py": "merlin/python/merlin/targetgen/_capsule_bundle_worker.py",
        "src/merlin/evaluation/capsule/certificates.py": "merlin/python/merlin/targetgen/capsule_runner.py",
        "src/merlin/evaluation/capsule/cpu.py": "merlin/python/merlin/targetgen/capsule_runner.py",
        "src/merlin/evaluation/capsule/diagnostics.py": "merlin/python/merlin/targetgen/capsule_diagnostics.py",
        "src/merlin/evaluation/capsule/eligibility.py": "merlin/python/merlin/targetgen/capsule_runner.py",
        "src/merlin/evaluation/capsule/identity.py": "merlin/python/merlin/targetgen/capsule_runner.py",
        "src/merlin/evaluation/capsule/inputs.py": "merlin/python/merlin/targetgen/capsule_runner.py",
        "src/merlin/evaluation/capsule/instructions.py": "merlin/python/merlin/targetgen/capsule_instruction_policy.py",
        "src/merlin/evaluation/capsule/ladder.py": "merlin/python/merlin/targetgen/capsule_runner.py",
        "src/merlin/evaluation/capsule/lanes.py": "merlin/python/merlin/targetgen/capsule_runner.py",
        "src/merlin/evaluation/capsule/lanes_evidence.py": "merlin/python/merlin/targetgen/capsule_lane_evidence.py",
        "src/merlin/evaluation/capsule/model.py": "merlin/python/merlin/targetgen/capsule_runner.py",
        "src/merlin/evaluation/capsule/model_assessment.py": "merlin/python/merlin/targetgen/capsule_runner.py",
        "src/merlin/evaluation/capsule/model_candidates.py": "merlin/python/merlin/targetgen/capsule_runner.py",
        "src/merlin/evaluation/capsule/model_context.py": "merlin/python/merlin/targetgen/capsule_candidate_context.py",
        "src/merlin/evaluation/capsule/model_evidence.py": "merlin/python/merlin/targetgen/capsule_runner.py",
        "src/merlin/evaluation/capsule/model_inputs.py": "merlin/python/merlin/targetgen/capsule_runner.py",
        "src/merlin/evaluation/capsule/model_runtime.py": "merlin/python/merlin/targetgen/capsule_runner.py",
        "src/merlin/evaluation/capsule/model_suite.py": "merlin/python/merlin/targetgen/capsule_runner.py",
        "src/merlin/evaluation/capsule/model_tiers.py": "merlin/python/merlin/targetgen/capsule_runner.py",
        "src/merlin/evaluation/capsule/numerics.py": "merlin/python/merlin/targetgen/capsule_runner.py",
        "src/merlin/evaluation/capsule/operation.py": "merlin/python/merlin/targetgen/capsule_runner.py",
        "src/merlin/evaluation/capsule/oracle_call.py": "merlin/python/merlin/targetgen/capsule_runner.py",
        "src/merlin/evaluation/capsule/oracle_result.py": "merlin/python/merlin/targetgen/capsule_runner.py",
        "src/merlin/evaluation/capsule/processes.py": "merlin/python/merlin/targetgen/capsule_processes.py",
        "src/merlin/evaluation/capsule/publication.py": "merlin/python/merlin/targetgen/capsule_runner.py",
        "src/merlin/evaluation/capsule/records.py": "merlin/python/merlin/targetgen/capsule_runner.py",
        "src/merlin/evaluation/capsule/reference.py": "merlin/python/merlin/targetgen/capsule_runner.py",
        "src/merlin/evaluation/capsule/result.py": "merlin/python/merlin/targetgen/capsule_runner.py",
        "src/merlin/evaluation/capsule/runner.py": "merlin/python/merlin/targetgen/capsule_runner.py",
        "src/merlin/evaluation/capsule/source.py": "merlin/python/merlin/targetgen/capsule_source_identity.py",
        "src/merlin/evaluation/capsule/suite.py": "merlin/python/merlin/targetgen/capsule_runner.py",
        "src/merlin/evaluation/capsule/tier.py": "merlin/python/merlin/targetgen/capsule_tier_result.py",
        "src/merlin/evaluation/capsule/trace.py": "merlin/python/merlin/targetgen/capsule_runner.py",
        "src/merlin/capture/artifacts.py": "merlin/python/merlin/targetgen/capsule_source.py",
        "src/merlin/capture/receipts.py": "merlin/python/merlin/targetgen/capsule_source.py",
        "src/merlin/capture/loaders.py": "merlin/python/merlin/targetgen/capsule_source.py",
        "src/merlin/capture/pytorch.py": "merlin/python/merlin/targetgen/capsule_source.py",
        "src/merlin/capture/cachekey.py": "merlin/python/merlin/targetgen/capsule_source.py",
        "src/merlin/capture/numerics.py": "merlin/python/merlin/targetgen/capsule_source.py",
        "src/merlin/capture/carriers.py": "merlin/python/merlin/targetgen/capsule_source.py",
        "src/merlin/capture/demands.py": "merlin/python/merlin/targetgen/capsule_source.py",
        "src/merlin/capture/spec.py": "merlin/python/merlin/targetgen/capsule_source.py",
        "src/merlin/experiments/corpus/source_evidence.py": "merlin/python/merlin/targetgen/capsule_source.py",
        "src/merlin/experiments/corpus/pytorch_sources.py": "merlin/python/merlin/targetgen/capsule_source.py",
        "src/merlin/experiments/corpus/model_inputs.py": "merlin/python/merlin/targetgen/capsule_source.py",
        "src/merlin/experiments/corpus/model_requirements.py": "merlin/python/merlin/targetgen/capsule_source.py",
        "src/merlin/experiments/corpus/model_sources.py": "merlin/python/merlin/targetgen/capsule_source.py",
        "src/merlin/experiments/corpus/spec_sources.py": "merlin/python/merlin/targetgen/capsule_source.py",
        "src/merlin/analysis/dse/guidance/insights/facts.py": "merlin/python/merlin/dse_guidance/insight_mining.py",
        "src/merlin/analysis/dse/guidance/insights/workloads.py": "merlin/python/merlin/dse_guidance/insight_mining.py",
        "src/merlin/analysis/dse/guidance/insights/influence.py": "merlin/python/merlin/dse_guidance/insight_mining.py",
        "src/merlin/analysis/dse/guidance/insights/audit.py": "merlin/python/merlin/dse_guidance/insight_mining.py",
        "src/merlin/analysis/dse/guidance/insights/findings.py": "merlin/python/merlin/dse_guidance/insight_mining.py",
        "src/merlin/analysis/dse/guidance/insights/emit.py": "merlin/python/merlin/dse_guidance/insight_mining.py",
        "src/merlin/analysis/dse/guidance/insights/evidence.py": "merlin/python/merlin/dse_guidance/insight_mining.py",
        "src/merlin/analysis/dse/guidance/insights/requirements.py": "merlin/python/merlin/dse_guidance/insight_mining.py",
        "src/merlin/analysis/dse/guidance/insights/primitives.py": "merlin/python/merlin/dse_guidance/insight_mining.py",
        "src/merlin/analysis/dse/guidance/insights/artifacts.py": "merlin/python/merlin/dse_guidance/insight_mining.py",
        "src/merlin/analysis/dse/guidance/insights/omissions.py": "merlin/python/merlin/dse_guidance/insight_mining.py",
        "src/merlin/analysis/dse/guidance/insights/plots.py": "merlin/python/merlin/dse_guidance/insight_mining.py",
        "src/merlin/analysis/dse/guidance/insights/corpus_facts.py": "merlin/python/merlin/dse_guidance/insight_mining.py",
        "src/merlin/analysis/dse/guidance/insights/summaries.py": "merlin/python/merlin/dse_guidance/insight_mining.py",
        "src/merlin/analysis/dse/guidance/insights/necessity.py": "merlin/python/merlin/dse_guidance/insight_mining.py",
        "src/merlin/analysis/dse/guidance/insight_mining.py": "merlin/python/merlin/dse_guidance/insight_mining.py",
        "src/merlin/analysis/dse/guidance/insights/capture.py": "merlin/python/merlin/dse_guidance/insight_mining.py",
        "src/merlin/compiler/optimizations/blocks.py": "merlin/python/merlin/llvmlower/impr_features.py",
        "src/merlin/compiler/optimizations/contraction.py": "merlin/python/merlin/llvmlower/impr_features.py",
        "src/merlin/compiler/optimizations/discovery.py": "merlin/python/merlin/llvmlower/impr_features.py",
        "src/merlin/compiler/optimizations/hoisting.py": "merlin/python/merlin/llvmlower/impr_features.py",
        "src/merlin/compiler/optimizations/loops.py": "merlin/python/merlin/llvmlower/impr_features.py",
        "src/merlin/compiler/optimizations/passes.py": "merlin/python/merlin/llvmlower/impr_features.py",
        "src/merlin/compiler/optimizations/perop.py": "merlin/python/merlin/llvmlower/impr_features.py",
        "src/merlin/compiler/optimizations/pointwise.py": "merlin/python/merlin/llvmlower/impr_features.py",
        "src/merlin/compiler/optimizations/registers.py": "merlin/python/merlin/llvmlower/impr_features.py",
        "src/merlin/compiler/optimizations/resident.py": "merlin/python/merlin/llvmlower/impr_features.py",
        "src/merlin/compiler/optimizations/variants.py": "merlin/python/merlin/llvmlower/impr_features.py",
        "src/merlin/compiler/optimizations/registry.py": "merlin/python/merlin/llvmlower/impr_features.py",
    }
    if path in relocated:
        return relocated[path]
    # This issuer predates the later merlin_experiments namespace. Its recorded
    # ceiling remains bound to the original shared-namespace implementation.
    if path in {
        "src/merlin/experiments/capture_execution/sealed_m2m.py",
        "packages/merlin-experiments/src/merlin/experiments/capture_execution/sealed_m2m.py",
    }:
        return "packages/merlin-experiments/src/merlin/experiments/capture_execution/sealed_m2m.py"
    # Historical ledgers used both physical spellings of this same research
    # owner. Normalize their shared-namespace spelling before generic core paths.
    historical = "packages/merlin-experiments/src/merlin/experiments"
    if path == historical or path.startswith(historical + "/"):
        return "packages/merlin-experiments/src/merlin_experiments" + path[len(historical) :]
    # Declaration/catalog orchestration moved intact from the research namespace.
    # Preserve its original policy identity, rather than granting compiler code
    # permission to read experiment definitions because it now shares a wheel.
    if path in {"src/merlin/development/definition.py", "src/merlin/experiments/definition.py"}:
        return "packages/merlin-experiments/src/merlin_experiments/spec.py"
    if path == "src/merlin/experiments" or path.startswith("src/merlin/experiments/"):
        return "packages/merlin-experiments/src/merlin_experiments" + path[len("src/merlin/experiments") :]
    for owner, historical in (("search", "dse"), ("pressure", "design_pressure"), ("guidance", "dse_guidance")):
        prefix = "src/merlin/analysis/dse/" + owner
        if path == prefix or path.startswith(prefix + "/"):
            return "merlin/python/merlin/" + historical + path[len(prefix) :]
    if path == "src/merlin/analysis/dse" or path.startswith("src/merlin/analysis/dse/"):
        return "packages/merlin-dse/src/merlin_dse" + path[len("src/merlin/analysis/dse") :]
    for canonical, former in (
        ("experiments/benchmarks", "merlin/benchmarks"),
        ("src/merlin/resources/benchmarks", "merlin/benchmarks"),
        ("src/merlin/resources/contracts", "merlin/contract"),
        ("src/merlin/resources/schemas", "merlin/schemas"),
        ("src/merlin/resources/prompts", "merlin/prompts"),
        ("src/merlin/runtime/native", "merlin/runtime"),
    ):
        if path == canonical or path.startswith(canonical + "/"):
            return former + path[len(canonical) :]
    if path == "src/merlin" or path.startswith("src/merlin/"):
        return "merlin/python/merlin" + path[len("src/merlin") :]
    for component in ("experiments", "analysis", "dse", "mining"):
        namespace = "src/merlin_" + component
        if path == namespace or path.startswith(namespace + "/"):
            return "packages/merlin-" + component + "/src/merlin_" + component + path[len(namespace) :]
    parts = path.split("/")
    if len(parts) >= 4 and parts[0] == "packages" and parts[2:4] == ["src", "merlin"]:
        return "/".join(("merlin", "python", "merlin", *parts[4:]))
    return path


def core_package(root: Path) -> Path:
    modern = root / "src/merlin"
    return modern if modern.is_dir() else root / "merlin/python/merlin"


def source_packages(root: Path) -> tuple[Path, ...]:
    """Package directories, preferring canonical src over its legacy compatibility symlink."""
    candidates = [*sorted((root / "src").glob("*")), root / "merlin/python/merlin"]
    candidates.extend(sorted((root / "tools/timing/src").glob("*")))
    candidates.extend(sorted((root / "packages").glob("*/src/*")))
    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        if path.is_dir() and path.name.isidentifier() and path.resolve() not in seen:
            seen.add(path.resolve())
            out.append(path)
    return tuple(out)


def logical_components(root: Path) -> dict:
    """Reviewed research boundaries; ordinary source placement grants no access."""

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate logical component key: {key}")
            result[key] = value
        return result

    path = root / "build_tools/logical_components.json"
    document = json.loads(path.read_text(), object_pairs_hook=unique)
    expected = {"merlin-experiments", "merlin-dse", "merlin-mining", "merlin-analysis"}
    if (
        set(document) != {"schema", "research"}
        or document["schema"] != "merlin.logical_components.v1"
        or type(document["research"]) is not dict
        or set(document["research"]) != expected
    ):
        raise ValueError("missing or unknown logical component inventory")
    seen = {}
    for owner, fields in document["research"].items():
        if type(fields) is not dict or set(fields) != {"namespaces", "modules"}:
            raise ValueError(f"invalid logical component fields: {owner}")
        for kind, values in fields.items():
            if type(values) is not list or len(values) != len(set(values)):
                raise ValueError(f"invalid logical component {kind}: {owner}")
            for value in values:
                if (
                    type(value) is not str
                    or not value
                    or any(not part.isidentifier() for part in value.split("."))
                    or value in seen
                ):
                    raise ValueError(f"invalid or overlapping logical component member: {value}")
                seen[value] = owner
    namespaces = [(name, owner) for owner, fields in document["research"].items() for name in fields["namespaces"]]
    if any(name != other and name.startswith(other + ".") for name, _ in namespaces for other, _ in namespaces):
        raise ValueError("overlapping logical component namespaces")
    if any(
        module.startswith(namespace + ".") or module == namespace
        for owner, fields in document["research"].items()
        for module in fields["modules"]
        for namespace, _ in namespaces
    ):
        raise ValueError("overlapping logical component module and namespace")
    return document["research"]


def component_owner(module: str, components: dict) -> str:
    for owner, fields in components.items():
        if module in fields["modules"] or any(
            module == prefix or module.startswith(prefix + ".") for prefix in fields["namespaces"]
        ):
            return owner
    return "core"


def module_name(path: Path, root: Path) -> str | None:
    """Import identity of source without importing core or an optional extension."""
    # Include both lexical core paths: callers may be inspecting historical grant/ledger entries.
    packages = (*source_packages(root), root / "src/merlin", root / "merlin/python/merlin")
    for package in packages:
        try:
            relative = path.relative_to(package.parent)
        except ValueError:
            continue
        if not relative.parts or relative.parts[0] != package.name:
            continue
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts)
    return None


def in_scope(rel: str, roots: tuple[str, ...], *, root: Path | None = None) -> bool:
    if not any(rel == scope or rel.startswith(scope + "/") for scope in roots):
        return False
    parts = Path(rel).parts
    for index, part in enumerate(parts):
        if part in EXCLUDED_PARTS:
            # compiler/build is authored Python, not an output directory.
            # Other generated build/dist trees remain pruned before descent.
            package = root is not None and (root.joinpath(*parts[: index + 1]) / "__init__.py").is_file()
            if part not in {"build", "dist"} or not package:
                return False
    # Extension-owned tests/resources are not library code; their own gates inspect them separately.
    if parts and parts[0] == "packages":
        return len(parts) >= 5 and parts[2] == "src"
    return True


def python_files(root: Path, roots: tuple[str, ...]) -> list[Path]:
    """Return repo-relative files, pruning generated trees before descent and deduplicating aliases."""
    candidates = []
    for scope in roots:
        if scope == "packages":
            candidates.extend(path for path in source_packages(root) if "packages" in path.relative_to(root).parts)
        else:
            candidates.append(root / scope)
    out: list[Path] = []
    seen: set[Path] = set()
    for base in candidates:
        for directory, subdirs, files in os.walk(base):
            subdirs[:] = sorted(
                name
                for name in subdirs
                if not name.startswith(".")
                and (
                    name not in EXCLUDED_PARTS
                    or name in {"build", "dist"}
                    and (Path(directory) / name / "__init__.py").is_file()
                )
            )
            for name in sorted(files):
                path = Path(directory) / name
                rel = path.relative_to(root)
                if not name.endswith(".py") or not in_scope(rel.as_posix(), roots, root=root) or path.resolve() in seen:
                    continue
                seen.add(path.resolve())
                out.append(rel)
    return out


def scan_python_paths(root: Path, roots: tuple[str, ...], *, staged: bool) -> list[Path]:
    if not staged:
        return python_files(root, roots)
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [Path(rel) for rel in result.stdout.splitlines() if rel.endswith(".py") and in_scope(rel, roots, root=root)]
