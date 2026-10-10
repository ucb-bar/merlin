"""Nothing a gemmini Phase 1 candidate can read names a prior implementation or its design choices.

A from-scratch compiler experiment is only from scratch if its inputs carry no answer. Two surfaces
reach the candidate: the repo paths its regenerated input bundle grants (exactly what
``corpus prepare`` scaffolds, :func:`merlin.targetgen.generate_bundles.generate_bundles`), and the
task brief rendered for it (:func:`merlin.targetgen.generate_prompt.render_prompt` plus the
launch-time blocks the Phase 1 stager appends). Both are scanned here for a deny-list of terms that
only appear when text describes a particular prior compiler, its support package, its runtime
conventions or its results.

Paths outside the checkout (the LLVM install, generated ``out/`` artifacts, external fact bundles)
are not repo text and are not scanned. A denied path stays unscanned: the sandbox masks it even
inside a granted directory.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from merlin.common.paths import python_source_dir, repo_root
from merlin.targetgen import tool_registry as TR
from merlin.targetgen.generalization_prompt import append_general_compiler_contract
from merlin.targetgen.generate_bundles import _ALL_ARMS, generate_bundles  # noqa: PLC2701
from merlin.targetgen.generate_prompt import render_prompt
from merlin.targetgen.sandbox.bwrap import resolve_grant
from merlin.targetgen.target_experiment import load_capability_manifest, load_target_experiment

pytestmark = pytest.mark.target("gemmini")

_VARIANTS = ("public_v0", "realistic_v0", "hwbringup_v0")

#: Terms that name a prior implementation, its packaging, its runtime conventions or its results.
DENY = re.compile(
    r"handwritten|champion|gemmini-mlir|merlin-support|weight[-_ ]stationary|im2col_recipes|tiled_matmul"
    r"|1ba5615f|801eec3f|gemmini_xdsl_rtl_v0",
    re.IGNORECASE,
)

#: Brief text that would suggest how to lower rather than what to compute: a lowering technique, a
#: data-layout choice listed as something to derive, or a concrete accelerator command sequence. A
#: format illustration (``<CLASS> 0x10 4``) names no class and no address.
LOWERING_HINT = re.compile(r"im2col|tiling, dtypes|dtypes, im2col|\b(?:MVIN|MVOUT|PRELOAD|COMPUTE_\w+) (?:0x)?\d")

#: A granted CAPSULE file states what to compute, never how a prior lowering computed it: no capsule an
#: agent reads may name a lowering strategy, in its name, its attributes, its provenance tags or its prose.
#: (Older records and captures are still READ through :mod:`merlin.targetgen.legacy_labels`.)
CAPSULE_ROOT = "merlin/contract/capsules/"
CAPSULE_DENY = re.compile(r"im2col", re.IGNORECASE)

#: Reviewed occurrences that are NOT descriptions of a prior implementation, by repo path and term.
#: Each must still occur (a stale entry fails), so this list cannot silently grow a blind spot.
ALLOWED = {
    # The command-buffer params key the reference semantics evaluate (a public ABI field name).
    ("src/merlin/runtime/commandbuffer.py", "im2col_recipes"),
    # The package manifest's generic publication flag (merlin.targetgen.publish), not a result.
    ("merlin/contract/schemas/manifest.schema.json", "champion"),
    # Public upstream test-source provenance of hand-authored legacy capsules (capsule bytes are bound).
    ("merlin/contract/capsules/isa/A3_k_accumulation/capsule.yaml", "tiled_matmul"),
    ("merlin/contract/capsules/model_slices/GF1_softmax_bf16_pt/capsule.yaml", "tiled_matmul"),
    ("merlin/contract/capsules/model_slices/GF3_geglu_bf16_pt/capsule.yaml", "tiled_matmul"),
}

#: Not repo text: the toolchain install, generated artifacts.
_UNSCANNED_TOP = ("third_party", "out")


def _descriptor():
    return load_target_experiment(repo_root() / "examples/gemmini/target/descriptor.yaml")


def _manifests():
    te = _descriptor()
    for variant in _VARIANTS:
        yield from generate_bundles(
            te, variant=variant, arms=tuple(_ALL_ARMS), python_source_root=python_source_dir()
        ).values()


def _granted_files() -> dict[str, Path]:
    """Every repo file some regenerated gemmini bundle grants, by repo-relative path."""
    root = repo_root().resolve()
    files: dict[str, Path] = {}
    for manifest in _manifests():
        denied = [resolve_grant(e["path"], root).resolve() for e in manifest.get("denied") or []]
        for entry in manifest["allowed"]:
            grant = resolve_grant(entry["path"], root).resolve()
            if not grant.exists() or not grant.is_relative_to(root):
                continue
            rel_top = grant.relative_to(root).parts[:1]
            if rel_top and rel_top[0] in _UNSCANNED_TOP:
                continue
            for path in [grant] if grant.is_file() else sorted(p for p in grant.rglob("*") if p.is_file()):
                if "__pycache__" in path.parts or any(path == d or path.is_relative_to(d) for d in denied):
                    continue
                files[path.relative_to(root).as_posix()] = path
    return files


def _hits(text: str) -> set[str]:
    return {m.group(0).lower().replace("-", "_").replace(" ", "_") for m in DENY.finditer(text)}


def _normalized(term: str) -> str:
    return term.lower().replace("-", "_").replace(" ", "_")


def test_no_granted_file_names_a_prior_implementation():
    files = _granted_files()
    assert len(files) > 100, f"only {len(files)} granted files found; the scan would prove nothing"
    allowed = {(path, _normalized(term)) for path, term in ALLOWED}
    found: dict[str, set[str]] = {}
    for rel, path in files.items():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        hits = {h for h in _hits(text) if (rel, h) not in allowed}
        if hits:
            found[rel] = hits
    assert not found, "agent-visible repo text names a prior implementation:\n" + "\n".join(
        f"  {rel}: {sorted(terms)}" for rel, terms in sorted(found.items())
    )


def test_no_granted_capsule_file_names_a_lowering_strategy():
    files = {rel: path for rel, path in _granted_files().items() if rel.startswith(CAPSULE_ROOT)}
    assert len(files) > 100, f"only {len(files)} granted capsule files found; the scan would prove nothing"
    found = []
    for rel, path in sorted(files.items()):
        if CAPSULE_DENY.search(rel):
            found.append(f"{rel} (path)")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if CAPSULE_DENY.search(text):
            found.append(rel)
    assert not found, "granted capsule files name a lowering strategy:\n  " + "\n  ".join(found)


def test_every_allowance_is_still_needed():
    files = _granted_files()
    stale = []
    for rel, term in sorted(ALLOWED):
        path = files.get(rel)
        if path is None or _normalized(term) not in _hits(path.read_text(encoding="utf-8")):
            stale.append((rel, term))
    assert not stale, f"remove these allowances; the text no longer occurs in a granted file: {stale}"


@pytest.mark.parametrize("experiment", ["full", "realistic"])
@pytest.mark.parametrize("arm", ["raw_baseline", "merlin_assisted", "merlin_rtlchecks"])
def test_the_rendered_task_brief_names_no_prior_implementation(experiment, arm):
    te, manifest = _descriptor(), load_capability_manifest("gemmini")
    body = render_prompt(te, manifest, experiment, arm, granted_tools=set())
    # The launch-time blocks the Phase 1 stager appends: the tool inventory and the general contract.
    body += "\n".join(f"- `{name}`: {TR.spec(name).blurb}" for name in TR.known_tools())
    body += "\n" + " ".join(filename for _, filename in TR.COMMON_CLIENTS)
    body = append_general_compiler_contract(body)
    realistic = te.resource_path("task/TASK_realistic.md")
    if experiment == "realistic" and realistic.is_file():
        body += realistic.read_text(encoding="utf-8")
    assert not _hits(body), f"the {experiment}/{arm} brief names: {sorted(_hits(body))}"
    hints = sorted({m.group(0) for m in LOWERING_HINT.finditer(body)})
    assert not hints, f"the {experiment}/{arm} brief suggests a lowering: {hints}"


@pytest.mark.parametrize("name", ["TASK.md", "TASK_full.md", "TASK_pilot.md", "TASK_realistic.md"])
def test_the_authored_task_briefs_suggest_no_lowering(name):
    path = _descriptor().resource_path(f"task/{name}")
    if not path.is_file():
        pytest.skip(f"{name} is not authored for this target")
    hints = sorted({m.group(0) for m in LOWERING_HINT.finditer(path.read_text(encoding="utf-8"))})
    assert not hints, f"{name} suggests a lowering: {hints}"


def test_the_fresh_component_author_prompt_names_no_prior_implementation():
    from merlin_experiments.phase1.component_origin import render_fresh_phase1_prompt

    prompt = render_fresh_phase1_prompt()
    assert not _hits(prompt), f"the fresh-author prompt names: {sorted(_hits(prompt))}"


def test_the_phase2_author_prompt_source_names_no_prior_implementation():
    """The Phase 2 brief is assembled from this module's text; none of it may name a prior implementation."""
    from merlin_experiments.phase2 import prompt

    text = Path(prompt.__file__).read_text(encoding="utf-8")
    assert not _hits(text), f"the Phase 2 prompt module names: {sorted(_hits(text))}"
