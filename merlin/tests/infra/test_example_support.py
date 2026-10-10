"""In-repo target support: recorded ownership, default selection, and agent-private bytes.

Each target's Merlin support provider is tracked at ``examples/<example>/support``, with a ``SOURCE.yaml``
beside it naming either canonical example ownership or a historical companion snapshot. Everything
here is derived from those records and providers' own declarations, so no target is named.

Three properties are held:

* canonical examples bind their current tracked tree; historical snapshots bind their companion tree,
  except the explicit path-normalized files whose original blobs reproduce that tree;
* with ``MERLIN_TARGET_PATH`` unset, each target's executable support is its in-repo provider, and any
  explicit value -- the empty string included -- replaces that default;
* support trees stay experimenter-side: a sandbox that exposes the whole checkout, a bundle snapshot
  or clean room built from a grant over all of ``examples/``, the transcript audit and publication all
  withhold or refuse them.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen import plugins, target_registry
from merlin.targetgen.providers import ProviderError, read_provider

RECORD = "SOURCE.yaml"
VENDORED_SCHEMA = "merlin.vendored_support.v1"
CANONICAL_SCHEMA = "merlin.canonical_example_support.v1"
#: A provider that is pure DATA (contract copy, plugin pointers, specs) served by generic core code.
GENERIC_SCHEMA = "merlin.generic_data_support.v1"
#: Members a data-only provider may hold. Anything else (code, headers, CRT, binaries) is refused.
GENERIC_DATA_SUFFIXES = {".yaml", ".yml", ".json", ".md"}


def _records() -> list[tuple[Path, dict]]:
    out = []
    for path in sorted((repo_root() / "examples").glob(f"*/{RECORD}")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        out.append((path, doc))
    return out


RECORDS = _records()
IDS = [path.parent.name for path, _ in RECORDS]
VENDORED_RECORDS = [(path, doc) for path, doc in RECORDS if doc.get("schema") == VENDORED_SCHEMA]


def _support_root(record: Path, doc: dict) -> Path:
    return (record.parent / doc["path"]).resolve()


def _tracked(root: Path) -> list[str]:
    """Tracked members of ``root``, relative to it (the index, so untracked caches never count)."""
    repo = repo_root()
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", str(root.relative_to(repo.resolve()))],
        cwd=repo,
        capture_output=True,
        check=True,
    ).stdout
    prefix = root.relative_to(repo.resolve()).as_posix() + "/"
    return sorted(item.decode()[len(prefix) :] for item in out.split(b"\0") if item)


def _git_tree_id(root: Path, members: list[str], blobs: dict[str, str] | None = None) -> str:
    """The git tree id ``members`` would have, computed from the bytes and modes on disk.

    ``blobs`` substitutes a recorded blob id for a member's bytes (same mode): the companion's blob for a
    file that was normalized when it was vendored.
    """
    blobs = blobs or {}
    tree: dict = {}
    for rel in members:
        node = tree
        *parents, leaf = rel.split("/")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = (root / rel, blobs.get(rel))

    def digest(node: dict) -> bytes:
        entries = []
        for name, value in node.items():
            if isinstance(value, dict):
                entries.append((name + "/", b"40000", name, digest(value)))
                continue
            path, recorded = value
            if path.is_symlink():
                mode, data = b"120000", os.readlink(path).encode()
            else:
                mode = b"100755" if path.stat().st_mode & 0o100 else b"100644"
                data = path.read_bytes()
            blob = bytes.fromhex(recorded) if recorded else hashlib.sha1(b"blob %d\0" % len(data) + data).digest()
            entries.append((name, mode, name, blob))
        entries.sort(key=lambda entry: entry[0].encode())
        body = b"".join(mode + b" " + name.encode() + b"\0" + sha for _, mode, name, sha in entries)
        return hashlib.sha1(b"tree %d\0" % len(body) + body).digest()

    return digest(tree).hex()


def test_every_in_repo_support_has_a_source_record():
    """No provider under ``examples/*/support`` without a record, and no record without its provider."""
    vendored = {root for root in target_registry.in_repo_support().values()}
    recorded = {_support_root(path, doc) for path, doc in RECORDS}
    assert vendored, "no vendored support provider was found; the default selection would be empty"
    assert vendored == recorded


def _assert_source_record(record: Path, doc: dict) -> None:
    root = _support_root(record, doc)
    members = _tracked(root)
    assert len(members) == doc["file_count"]
    if doc.get("schema") == VENDORED_SCHEMA:
        assert _git_tree_id(root, members) == doc["vendored_tree"], "vendored bytes differ from the recorded tree"
        normalized = {entry["path"]: entry["companion_blob"] for entry in doc["normalized"]}
        assert set(normalized) <= set(members), "a normalized file is not part of the vendored tree"
        assert all(entry["change"].strip() for entry in doc["normalized"]), "a normalization states no change"
        # Only listed files may differ: their companion blobs reproduce the companion tree.
        assert _git_tree_id(root, members, normalized) == doc["source"]["tree"], "an unlisted file differs"
        assert (doc["vendored_tree"] == doc["source"]["tree"]) == (not normalized)
        if doc["pinned"]["identical_to_source"]:
            assert doc["pinned"]["provider_tree"] == doc["source"]["tree"]
    elif doc.get("schema") == CANONICAL_SCHEMA:
        assert set(doc) == {
            "schema",
            "target",
            "provider_id",
            "path",
            "role",
            "visibility",
            "external_support_checkout_required",
            "file_count",
            "canonical_since",
            "canonical_tree",
            "origin",
            "records",
            "tests",
        }
        assert doc["role"] == "support" and doc["visibility"] == "experimenter_only"
        assert doc["external_support_checkout_required"] is False
        assert _git_tree_id(root, members) == doc["canonical_tree"], "canonical support bytes changed"
        origin = doc["origin"]
        assert set(origin) == {"repository", "published", "commit", "commit_date", "provider_root", "tree"}
        assert origin["repository"].startswith("https://") and type(origin["published"]) is bool
        assert all(
            len(origin[key]) == 40 and all(ch in "0123456789abcdef" for ch in origin[key]) for key in ("commit", "tree")
        )
        datetime.fromisoformat(origin["commit_date"])
        assert origin["provider_root"] and not Path(origin["provider_root"]).is_absolute()
    elif doc.get("schema") == GENERIC_SCHEMA:
        assert set(doc) == {
            "schema",
            "target",
            "provider_id",
            "path",
            "role",
            "visibility",
            "file_count",
            "tree",
            "backend",
            "records",
            "experimenter_tools",
        }
        assert doc["role"] == "support" and doc["visibility"] == "experimenter_only"
        assert _git_tree_id(root, members) == doc["tree"], "data provider bytes changed"
        # Code may live only in declared experimenter-side tool directories (masked with the rest of the
        # provider), never in what the provider selects or serves.
        tools = tuple(f"{name.rstrip('/')}/" for name in doc["experimenter_tools"])
        assert all((root / name).is_dir() for name in tools), "a declared experimenter tool directory is missing"
        assert [m for m in members if Path(m).suffix not in GENERIC_DATA_SUFFIXES and not m.startswith(tools)] == [], (
            "a data provider holds code outside its declared experimenter tools"
        )
        plugin_refs = [
            str(v) for v in yaml.safe_load((root / "contracts/target_contract.yaml").read_text())["plugin"].values()
        ]
        assert not any(ref.startswith(tools) for ref in plugin_refs), "a plugin selects experimenter tooling"
        selected = yaml.safe_load((root / "contracts/target_contract.yaml").read_text(encoding="utf-8"))
        assert selected["plugin"]["backend"] == doc["backend"]
        assert plugins.core_module_path(doc["backend"]) is not None, "the backend is not generic core code"
    else:
        raise AssertionError("unknown support ownership schema")
    provider = read_provider(root)
    assert (provider.id, provider.target) == (doc["provider_id"], doc["target"])
    records = doc["records"]
    for relative in (records["file_provenance"], *records["migrations"]):
        assert (record.parent / relative).is_file(), relative


@pytest.mark.parametrize(("record", "doc"), RECORDS, ids=IDS)
def test_support_matches_its_recorded_ownership(record, doc):
    _assert_source_record(record, doc)


@pytest.mark.parametrize("mutation", ["tree", "count", "role", "external_pin"])
def test_canonical_support_record_refuses_mutation(mutation):
    # Test this metadata protocol without requiring a canonical reference backend
    # to remain installed. Derive a diagnostic record from an existing tracked
    # snapshot; it confers no runtime or fresh-experiment authority.
    record, snapshot = VENDORED_RECORDS[0]
    original = {
        "schema": CANONICAL_SCHEMA,
        "target": snapshot["target"],
        "provider_id": snapshot["provider_id"],
        "path": snapshot["path"],
        "role": "support",
        "visibility": "experimenter_only",
        "external_support_checkout_required": False,
        "file_count": snapshot["file_count"],
        "canonical_since": snapshot["vendored"],
        "canonical_tree": snapshot["vendored_tree"],
        "origin": snapshot["source"],
        "records": snapshot["records"],
        "tests": [],
    }
    _assert_source_record(record, original)
    doc = {**original}
    if mutation == "tree":
        doc["canonical_tree"] = "0" * 40
    elif mutation == "count":
        doc["file_count"] += 1
    elif mutation == "role":
        doc["role"] = "candidate"
    else:
        doc["pinned"] = {"companion_commit": "0" * 40}
    with pytest.raises(AssertionError):
        _assert_source_record(record, doc)


def test_the_migration_manifest_agrees_with_every_source_record():
    """The registry distinguishes historical companion pins from canonical current trees."""
    import json

    manifest = json.loads((repo_root() / "build_tools/upstreams/target_support.json").read_text(encoding="utf-8"))
    listed_legacy = {
        entry["vendored"]["source_record"]: entry for entry in manifest["companions"] if "vendored" in entry
    }
    listed_canonical = {
        entry["canonical_example"]["source_record"]: entry
        for entry in manifest["companions"]
        if "canonical_example" in entry
    }
    listed_generic = {
        entry["generic_data_support"]["source_record"]: entry
        for entry in manifest["companions"]
        if "generic_data_support" in entry
    }
    assert not (set(listed_legacy) & set(listed_canonical))
    assert not (set(listed_generic) & (set(listed_legacy) | set(listed_canonical)))
    assert set(listed_legacy) | set(listed_canonical) | set(listed_generic) == {
        path.relative_to(repo_root()).as_posix() for path, _ in RECORDS
    }
    for path, doc in RECORDS:
        key = path.relative_to(repo_root()).as_posix()
        if doc["schema"] == GENERIC_SCHEMA:
            entry = listed_generic[key]
            generic = entry["generic_data_support"]
            assert entry["ownership"] == "generic_data_support"
            assert set(generic) == {"path", "source_record", "tree", "file_count", "backend"}
            assert (generic["tree"], generic["file_count"], generic["backend"]) == (
                doc["tree"],
                doc["file_count"],
                doc["backend"],
            )
            assert (
                entry["provider_root"]
                == generic["path"]
                == _support_root(path, doc).relative_to(repo_root()).as_posix()
            )
            assert (entry["provider_role"], entry["target"]) == (doc["role"], doc["target"])
            assert "vendored" not in entry and "companion_commit" not in entry
            continue
        if doc["schema"] == CANONICAL_SCHEMA:
            entry = listed_canonical[key]
            canonical = entry["canonical_example"]
            assert set(canonical) == {"path", "source_record", "tree", "file_count"}
            assert entry["ownership"] == "canonical_example"
            assert entry["external_support_checkout_required"] is False
            assert (
                entry["provider_root"]
                == canonical["path"]
                == _support_root(path, doc).relative_to(repo_root()).as_posix()
            )
            assert (canonical["tree"], canonical["file_count"]) == (doc["canonical_tree"], doc["file_count"])
            assert all(entry["origin"][field] == value for field, value in doc["origin"].items())
            assert "vendored" not in entry and "companion_commit" not in entry
            assert "companion_commit_relation" not in entry
            assert (entry["provider_role"], entry["target"]) == (doc["role"], doc["target"])
            continue
        assert doc["schema"] == VENDORED_SCHEMA
        entry = listed_legacy[key]
        vendored = entry["vendored"]
        assert entry["target"] == doc["target"]
        assert (vendored["commit"], vendored["tree"], vendored["vendored_tree"], vendored["file_count"]) == (
            doc["source"]["commit"],
            doc["source"]["tree"],
            doc["vendored_tree"],
            doc["file_count"],
        )
        assert vendored.get("merge_parents") == doc["source"].get("merge_parents")
        assert entry["companion_commit"] == doc["pinned"]["companion_commit"]
        assert entry["companion_commit_relation"] == doc["pinned"]["relation"]


#: How the pinned companion commit relates to the commit the bytes were copied from.
PIN_RELATIONS = {"same_commit", "base_pin_of_merged_tip"}


@pytest.mark.parametrize(("record", "doc"), VENDORED_RECORDS, ids=[path.parent.name for path, _ in VENDORED_RECORDS])
def test_the_pinned_commit_and_the_copied_commit_are_told_apart(record, doc):
    """Two different commits in one record must say which is which, and must carry one provider tree.

    A merged tip (``source.commit``, with its ``merge_parents``) may be what the bytes were copied from
    while the manifest pins the support-branch commit it merged (the base pin). That is consistent only
    if the record names the relation and both commits carry the same provider tree; an unexplained
    second commit is the inconsistency this refuses.
    """
    source, pinned = doc["source"], doc["pinned"]
    relation = pinned["relation"]
    assert relation in PIN_RELATIONS
    if relation == "same_commit":
        assert pinned["companion_commit"] == source["commit"]
        assert "merge_parents" not in source
    else:
        assert pinned["companion_commit"] != source["commit"]
        assert len(source.get("merge_parents") or []) >= 2, "a merged tip names its parents"
        assert pinned["provider_tree"] == source["tree"], "the base pin and the merged tip carry one tree"
        when = datetime.fromisoformat
        assert when(pinned["companion_commit_date"]) <= when(source["commit_date"]), "a pin newer than its merge"


def test_a_second_commit_without_a_relation_is_refused():
    """The relation check above can fail: a record that pins a different commit as ``same_commit`` is caught."""
    path, doc = next((path, doc) for path, doc in VENDORED_RECORDS if doc["pinned"]["relation"] == "same_commit")
    forged = {**doc, "pinned": {**doc["pinned"], "companion_commit": "0" * 40}}
    with pytest.raises(AssertionError):
        test_the_pinned_commit_and_the_copied_commit_are_told_apart(path, forged)


@pytest.mark.parametrize(("record", "doc"), RECORDS, ids=IDS)
def test_unset_selection_is_the_targets_vendored_support(record, doc, monkeypatch):
    monkeypatch.delenv("MERLIN_TARGET_PATH", raising=False)
    root, target = _support_root(record, doc), doc["target"]
    assert target_registry.default_support_root(target) == root
    assert target_registry.explicit_targets()[target] == root
    resolved = target_registry.resolve(target)
    assert (resolved.kind, resolved.base) == ("external", root)
    assert plugins.resolve_support(target).base == root


def test_an_explicit_selection_replaces_the_default(tmp_path, monkeypatch):
    record, doc = RECORDS[0]
    target = doc["target"]
    monkeypatch.setenv("MERLIN_TARGET_PATH", "")
    assert target_registry.explicit_targets() == {}
    assert target_registry.effective_target_path() == ""
    # The refusal names the vendored provider the explicit value replaced, and how to get it back.
    with pytest.raises(plugins.PluginError, match="explicit MERLIN_TARGET_PATH") as refused:
        plugins.resolve_support(target)
    assert str(_support_root(record, doc)) in str(refused.value) and "unset the variable" in str(refused.value)

    elsewhere = tmp_path / "pinned-support"
    (elsewhere / "contracts").mkdir(parents=True)
    (elsewhere / "contracts/target_contract.yaml").write_text(f"name: {target}\n", encoding="utf-8")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(elsewhere))
    assert target_registry.explicit_targets() == {target: elsewhere.resolve()}
    assert target_registry.resolve(target).base == elsewhere.resolve()

    monkeypatch.delenv("MERLIN_TARGET_PATH")
    spelled = target_registry.effective_target_path().split(os.pathsep)
    assert spelled == [str(root) for root in target_registry.in_repo_support().values()]


def test_in_repo_support_is_keyed_by_declaration_not_directory(tmp_path, monkeypatch):
    def provider(example: str, target: str) -> Path:
        root = tmp_path / "examples" / example / target_registry.IN_REPO_SUPPORT_DIR
        (root / "contracts").mkdir(parents=True)
        (root / "contracts/target_contract.yaml").write_text(f"name: {target}\n", encoding="utf-8")
        (root / "provider.yaml").write_text(
            f"schema: merlin.provider.v1\nid: {example}-support\ntarget: {target}\nrole: support\n",
            encoding="utf-8",
        )
        return root

    monkeypatch.setattr(target_registry, "checkout_root", lambda: tmp_path)
    declared = provider("folder_name", "declared_device")
    (tmp_path / "examples/metadata_only/support").mkdir(parents=True)  # not a provider: ignored
    assert target_registry.in_repo_support() == {"declared_device": declared.resolve()}

    provider("second_folder", "declared_device")
    with pytest.raises(target_registry.TargetCollisionError, match="declared_device"):
        target_registry.in_repo_support()

    (tmp_path / "examples/second_folder/support/provider.yaml").write_text("schema: wrong\n", encoding="utf-8")
    with pytest.raises(ProviderError):
        target_registry.in_repo_support()

    monkeypatch.setattr(target_registry, "checkout_root", lambda: None)  # an installed distribution
    assert target_registry.in_repo_support() == {}


# ------------------------------------------------------------------------------------- the boundary
DESCRIPTORS = sorted((repo_root() / "examples").glob("*/target/descriptor.yaml"))


def _private_members(root: Path) -> list[Path]:
    return [root / rel for rel in _tracked(root)]


@pytest.fixture(scope="module")
def sandbox():
    pytest.importorskip("merlin_experiments")
    import importlib

    # The package re-exports a function named ``answer_surfaces``; the module is the one wanted.
    modules = {
        name: importlib.import_module(f"merlin.targetgen.sandbox.{name}")
        for name in ("answer_surfaces", "bwrap", "cleanroom")
    }
    return SimpleNamespace(surfaces=modules["answer_surfaces"], bwrap=modules["bwrap"], cleanroom=modules["cleanroom"])


@pytest.mark.parametrize("descriptor", DESCRIPTORS, ids=[d.parent.parent.name for d in DESCRIPTORS])
def test_a_sandbox_exposing_the_whole_checkout_shows_no_vendored_support(descriptor, sandbox):
    """Whichever target an arm works on, every vendored support tree is masked, contracts included.

    A real bundle grants specific paths, none under ``examples/*/support``; exposing the entire checkout
    is the worst case those grants could add up to.
    """
    from merlin.targetgen.target_experiment import load_target_experiment

    te = load_target_experiment(descriptor)
    derived = sandbox.surfaces.answer_surfaces(te)
    repo = str(repo_root())
    exposed = ["--ro-bind", repo, repo]
    roots = list(target_registry.in_repo_support().values())
    assert roots and all(any(item.path == root for item in derived) for root in roots)
    assert all(sandbox.bwrap.is_exposed(exposed, _private_members(root)[0]) for root in roots)
    masked = sandbox.bwrap.apply_answer_masks(exposed, derived)
    assert sandbox.bwrap.coverage_gap(masked, derived) == []
    leaked = [member for root in roots for member in _private_members(root) if sandbox.bwrap.is_exposed(masked, member)]
    assert leaked == []


def test_a_bundle_snapshot_of_all_examples_withholds_vendored_support(sandbox, tmp_path, monkeypatch):
    """The agent-visible snapshot: a grant over ``examples/`` freezes the support bytes as private views."""
    from merlin.targetgen.target_experiment import load_target_experiment

    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(tmp_path / "cas"))
    te = load_target_experiment(DESCRIPTORS[0])
    repo = repo_root().resolve()
    ws = tmp_path / "run/workspace"
    ws.mkdir(parents=True)
    bundle = {"allowed": [{"path": "examples/"}]}
    manifest = sandbox.bwrap.materialize_bundle_inputs(ws, bundle, repo=repo)
    frozen = sandbox.bwrap.snapshot_input_paths(ws, bundle, [repo / "examples"], repo=repo)[0]
    views = sandbox.bwrap.snapshot_support_surfaces(ws, manifest)
    view = tmp_path / "agent-view"
    argv = sandbox.bwrap.apply_final_answer_masks(["--ro-bind", str(frozen), str(view)], te, ws, bundle, repo=repo)
    for root in target_registry.in_repo_support().values():
        for member in _private_members(root):
            relative = member.relative_to(repo / "examples")
            assert not sandbox.bwrap.snapshot_public_member(frozen / relative, frozen, views), relative
            assert not sandbox.bwrap.is_exposed(argv, view / relative), relative


def test_a_clean_room_from_all_examples_places_no_vendored_support(sandbox, tmp_path):
    """Placement, judged by path: a grant over every example copies none of the vendored support.

    The verifier's content sieve is off here on purpose. Some public bytes are byte-identical to files
    inside a support tree (a curated harness header the support also carries), and the sieve refuses a
    room for that whichever tree the bytes were placed from. That stricter rule is the clean room's
    own policy and is unchanged; this test isolates the question of what the builder places.
    """
    from merlin.targetgen.target_experiment import load_target_experiment

    te = load_target_experiment(DESCRIPTORS[0])
    room = sandbox.cleanroom.build_clean_room(
        te, tmp_path / "home", {"allowed": [{"path": "examples/"}]}, check_content=False
    )
    placed = room.inputs / "repo/examples"
    assert placed.is_dir(), "the grant placed nothing, so this proves nothing"
    examples = repo_root().resolve() / "examples"
    vendored = [placed / root.relative_to(examples) for root in target_registry.in_repo_support().values()]
    # A surface that declares a grantable sub-tree is walked into, so its directory may exist, empty.
    leaked = [path for root in vendored if root.exists() for path in root.rglob("*") if not path.is_dir()]
    assert leaked == []


def test_the_transcript_audit_names_every_vendored_support_entry(sandbox):
    from merlin.targetgen.target_experiment import load_target_experiment

    tokens = sandbox.surfaces.audit_tokens(load_target_experiment(DESCRIPTORS[0]))["answer"]
    for root in target_registry.in_repo_support().values():
        for child in root.iterdir():
            if child.name != sandbox.surfaces.PACKAGE_CONTRACT_SUBDIR:
                assert any(token in str(child / "x") for token in tokens), child


@pytest.mark.parametrize(("record", "doc"), RECORDS, ids=IDS)
def test_vendored_support_cannot_be_published_as_a_candidate(record, doc):
    from merlin.targetgen import publish

    with pytest.raises(publish.PublishError, match="not support"):
        publish._export_role(SimpleNamespace(package_dir=_support_root(record, doc), target=doc["target"]))


# --------------------------------------------------------------------------------- vendored suites
SUITES = [(path, doc) for path, doc in RECORDS if (_support_root(path, doc) / "tests").is_dir()]
#: A suite's tests that need an operator's external checkout, recorded in SOURCE.yaml with the
#: checkout's ``MERLIN_EXT_<NAME>`` key and why: they run where it is set and skip, by name, where not.
NEEDS_EXTERNAL = [
    (path, doc, entry) for path, doc in SUITES for entry in ((doc.get("tests") or {}).get("requires_external") or [])
]


def _run_suite(root: Path, tmp_path: Path, selection: list[str]) -> subprocess.CompletedProcess:
    """A provider's own tests, run from its in-repo home the way its README documents.

    The README selects the provider alone on ``MERLIN_TARGET_PATH``; so does this, with the vendored
    root, so the suite sees exactly the plugins it saw at its companion. Each suite runs in its own
    interpreter because the suites import their provider's modules by bare name and share test-module
    basenames.
    """
    env = dict(os.environ, MERLIN_TARGET_PATH=str(root), PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(root), os.environ.get("PYTHONPATH"))))
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--rootdir={repo_root().resolve()}",
            f"--basetemp={tmp_path / 'basetemp'}",
            *selection,
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(("record", "doc"), SUITES, ids=[path.parent.name for path, _ in SUITES])
def test_vendored_support_suite_passes_from_its_new_home(record, doc, tmp_path):
    """Every vendored provider's suite, in the fast CI job: a core change that breaks a provider which is
    never edited in place has to fail somewhere.

    Two recorded exclusions, each with its reason in SOURCE.yaml rather than an edit to the vendored
    tree: a test bound to the companion repository's own layout (``tests.deselect``), and a test that
    needs an operator's external checkout (``tests.requires_external``), which runs in the test below.
    """
    root = _support_root(record, doc)
    repo = repo_root().resolve()
    tests = doc.get("tests") or {}
    excluded = [entry["test"] for entry in (*tests.get("deselect", []), *(tests.get("requires_external") or []))]
    assert all((root / relative.partition("::")[0]).is_file() for relative in excluded), "an exclusion names nothing"
    selection = [f"--deselect={(root / relative).relative_to(repo).as_posix()}" for relative in excluded]
    result = _run_suite(root, tmp_path, [*selection, str(root / "tests")])
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]


@pytest.mark.parametrize(
    ("record", "doc", "entry"),
    NEEDS_EXTERNAL,
    ids=[f"{path.parent.name}-{entry['external']}" for path, _, entry in NEEDS_EXTERNAL],
)
def test_vendored_support_tests_that_need_an_external_checkout(record, doc, entry, tmp_path):
    from merlin.common.paths import ExternalPathUnset, ext_path

    try:
        ext_path(entry["external"])
    except ExternalPathUnset:
        pytest.skip(f"MERLIN_EXT_{entry['external'].upper()} is unset: {' '.join(entry['reason'].split())}")
    root = _support_root(record, doc)
    result = _run_suite(root, tmp_path, [str(root / entry["test"])])
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]


def test_a_support_directory_without_a_declaration_is_still_masked(sandbox, tmp_path, monkeypatch):
    """Selection counts declared providers; the mask must not. A ``support/`` tree with no
    ``provider.yaml`` is still target support bytes."""
    undeclared = tmp_path / "examples" / "draft" / target_registry.IN_REPO_SUPPORT_DIR
    (undeclared / "backend").mkdir(parents=True)
    monkeypatch.setattr(target_registry, "checkout_root", lambda: tmp_path)
    assert target_registry.in_repo_support() == {}
    assert target_registry.vendored_support_dirs() == (undeclared.resolve(),)
    monkeypatch.undo()
    monkeypatch.setattr(target_registry, "vendored_support_dirs", lambda: (undeclared.resolve(),))
    assert undeclared.resolve() in sandbox.surfaces._support_package_dirs()


def test_exposing_the_checkout_does_not_expose_its_git_store(sandbox):
    """``.git`` holds every tracked byte the masks withhold, the vendored support included."""
    from merlin.targetgen.target_experiment import load_target_experiment

    derived = sandbox.surfaces.answer_surfaces(load_target_experiment(DESCRIPTORS[0]))
    repo = repo_root()
    store = repo / ".git"
    assert store in [item.path for item in derived if item.origin == "git"]
    masked = sandbox.bwrap.apply_answer_masks(["--ro-bind", str(repo), str(repo)], derived)
    assert sandbox.bwrap.is_exposed(["--ro-bind", str(repo), str(repo)], store)
    assert not sandbox.bwrap.is_exposed(masked, store)


def test_the_transcript_audit_names_a_support_read_made_from_inside_examples(sandbox):
    """``cd examples && cat <example>/support/...`` carries no ``examples/`` prefix; it is still a read."""
    from merlin.targetgen.target_experiment import load_target_experiment

    tokens = sandbox.surfaces.audit_tokens(load_target_experiment(DESCRIPTORS[0]))["answer"]
    for root in target_registry.in_repo_support().values():
        for child in root.iterdir():
            if child.name != sandbox.surfaces.PACKAGE_CONTRACT_SUBDIR:
                assert f"{root.parent.name}/{root.name}/{child.name}" in tokens, child
