"""The recorded performance-template provenance must name a file that actually exists.

`generate_corpus.py` stamps every generated `_perf` corpus with the path and digest of the one
shared template it expanded (`profiles/_perf.yaml`), and `perf_campaign.discover_performance_corpus`
refuses the corpus unless that record resolves and still matches the template on disk. Both halves
went silently wrong: the writer rooted the relative path at `<repo>/merlin` instead of `<repo>`, so
the recorded `contract/capsules/profiles/_perf.yaml` missed the post-reorg `merlin/` prefix and the
gate raised `required campaign evidence is absent` against its own corpus.

Nothing compared the record to the file, so the miss survived a reorg and several template edits.
These tests do that comparison directly.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys

import pytest
import yaml

from merlin.common.paths import repo_root

CAPSULES = repo_root() / "merlin" / "contract" / "capsules"
MANIFEST = CAPSULES / "MANIFEST.yaml"


def _generator():
    """Load `generate_corpus.py` by path -- it is a script, not an importable package member."""
    path = CAPSULES / "generate_corpus.py"
    if str(CAPSULES) not in sys.path:
        sys.path.insert(0, str(CAPSULES))
    spec = importlib.util.spec_from_file_location("generate_corpus_provenance_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _document_digest(document: object) -> str:
    """The digest both the generator and the campaign gate agree on: over the PARSED document."""
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _check_record(record: dict, *, owner: str) -> None:
    path = record.get("path")
    assert isinstance(path, str) and path, f"{owner}: shared_template.path is missing"
    resolved = repo_root() / path
    assert resolved.is_file(), (
        f"{owner}: shared_template.path {path!r} does not resolve to a file from repo_root() "
        f"({resolved}). A repo-root-relative record must be spelled from the repo root, "
        "`merlin/` prefix included.")
    document = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    assert record.get("sha256") == _document_digest(document), (
        f"{owner}: recorded template digest is stale relative to {path}; the corpus was generated "
        "from a materially older template. Regenerate it.")


def test_generator_records_a_template_path_that_resolves_from_repo_root():
    """The WRITER's contract: what it stamps must resolve, and its digest must match the file.

    This is the regression guard. It fails when the generator roots the relative path anywhere but
    the repository root -- the exact defect that made the campaign gate refuse its own corpus.
    """
    template = _generator().load_profile("gemmini")["_performance_template"]
    assert not template["path"].startswith("/"), "the record must be relative, not absolute"
    _check_record(template, owner="generate_corpus.load_profile('gemmini')")


def test_generator_records_every_family_the_template_declares():
    """A stale families list is the other way this record silently drifts from the template."""
    generator = _generator()
    template = generator.load_profile("gemmini")["_performance_template"]
    document = yaml.safe_load((repo_root() / template["path"]).read_text(encoding="utf-8"))
    declared = {str(sweep["id"]) for sweep in (document.get("sweeps") or [])}
    declared |= {str(row["family"]) for row in (document.get("blocked_unimplemented") or [])}
    assert {str(row["family"]) for row in template["families"]} == declared


@pytest.mark.xfail(strict=True, reason=(
    "MANIFEST.yaml still records the pre-fix, `merlin/`-less template path, because it was last "
    "written by the generator before this fix. Regenerating it in place was held back while a "
    "gemmini functional run was grading against merlin/contract/capsules/. Re-run "
    "`.venv/bin/python merlin/contract/capsules/generate_corpus.py --target gemmini` (and again "
    "with `--target atlas`) once that run finishes, then delete this marker."))
def test_tracked_manifest_template_provenance_is_current():
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    generation = manifest.get("performance_generation") or {}
    assert generation, "MANIFEST.yaml records no performance_generation provenance"
    for target, record in sorted(generation.items()):
        _check_record(record["shared_template"], owner=f"MANIFEST.yaml performance_generation.{target}")
