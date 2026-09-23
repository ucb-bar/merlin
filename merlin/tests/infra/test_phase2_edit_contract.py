"""The declared edit surface validates, and the two ways it can be wrong are refused by name.

The two failures are not symmetric. A contract naming a symbol that does not exist is loud -- it
raises on the first validation. A contract OMITTING the symbol that decides is silent: it validates,
it runs, it produces rounds and a verdict, and the experiment was unwinnable the whole time. These
tests exist mostly for the second one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.perf.compiler_edit_scope import inspect_compiler_edits
from merlin.perf.phase2_edit_contract import (
    Phase2EditContractError,
    contract_path,
    load,
    seal,
    validate_against_package,
)

# The declarations this repo ships, as (target, package_id). The target is a directory component --
# the sanctioned edge for a target name -- and neither string appears in library code.
DECLARED = [("gemmini", "gemmini_xdsl_rtl_v0"), ("gemmini", "gemmini_xdsl_oot_v1_epilogue")]


def _body(target: str, package_id: str) -> dict:
    path = contract_path(target, package_id)
    if not path.is_file():
        pytest.fail(f"declared contract missing at {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _package_root(body: dict) -> Path:
    return repo_root() / body["package"]["root"]


@pytest.mark.parametrize("target,package_id", DECLARED)
def test_declaration_loads_and_seals(target, package_id):
    body = _body(target, package_id)
    contract = load(target, package_id)
    assert contract["schema"] == "compiler_edit_contract_v1"
    # The digest is over the body, recomputed -- so `validate_edit_contract`, which recomputes it
    # independently, cannot disagree with us about the contract's identity.
    assert contract["sha256"] == seal(body)["sha256"]
    assert len(contract["sha256"]) == 64


@pytest.mark.parametrize("target,package_id", DECLARED)
def test_validates_against_the_real_package(target, package_id):
    """Every named symbol resolves in the package's own AST.

    Skipped rather than failed when the package is absent: these are tool-generated artifacts and
    only the hand-authored baselines are tracked, so a fresh clone or a worktree legitimately has
    neither. A skip says "not checked here"; a pass would say "checked and fine", which would be
    the hollow positive this whole apparatus is written against.
    """
    body = _body(target, package_id)
    root = _package_root(body)
    if not root.is_dir():
        pytest.skip(f"package not present at {root} (tool-generated; not tracked)")
    resolved = validate_against_package(load(target, package_id), root)
    assert resolved["existing_symbols"]


@pytest.mark.parametrize("target,package_id", DECLARED)
def test_contract_root_is_where_the_manifest_is(target, package_id):
    """`inspect_compiler_edits` protects `manifest.yaml` ONLY at the contract root.

    One of these packages keeps its manifest a level below the package directory. Rooting the
    contract at the package directory would leave the manifest an ordinary file and the
    self-authorization protection silently off, with the contract looking identical.
    """
    body = _body(target, package_id)
    root = _package_root(body)
    if not root.is_dir():
        pytest.skip(f"package not present at {root}")
    assert (root / "manifest.yaml").is_file(), (
        f"{target}/{package_id}: no manifest.yaml at the declared contract root {root}; the "
        "protection against a candidate widening its own authority would not fire"
    )


@pytest.mark.parametrize("target,package_id", DECLARED)
def test_omitting_a_required_owner_is_refused_by_name(target, package_id):
    """The silent failure, made loud. Each required decision, dropped one at a time."""
    body = _body(target, package_id)
    for required in body["required_decisions"]:
        owner, decision = required["owner"], required["decision"]
        mutated = dict(body)
        mutated["existing_symbols"] = [
            entry for entry in body["existing_symbols"] if f"{entry['path']}:{entry['symbol']}" != owner
        ]
        assert len(mutated["existing_symbols"]) < len(body["existing_symbols"]), (
            f"{decision}: owner {owner} is not in existing_symbols, so this contract already fails"
        )
        with pytest.raises(Phase2EditContractError) as caught:
            load(target, package_id, body=mutated)
        message = str(caught.value)
        assert decision in message, f"refusal does not name the decision {decision!r}: {message}"
        assert owner in message, f"refusal does not name the owner {owner!r}: {message}"


def test_declaration_without_required_decisions_is_refused():
    """A contract that declares nothing to be reachable cannot show anything is reachable."""
    body = {
        "schema": "compiler_edit_contract_v1",
        "existing_symbols": [{"surface_id": "s", "path": "a.py", "symbol": "f"}],
    }
    with pytest.raises(Phase2EditContractError, match="required_decisions"):
        load("t", "p", body=body)


def test_a_symbol_absent_from_the_package_is_refused(tmp_path):
    """The loud failure, for completeness: a named symbol that does not exist."""
    (tmp_path / "compiler").mkdir()
    (tmp_path / "compiler/codegen.py").write_text("def present():\n return 1\n")
    body = {
        "schema": "compiler_edit_contract_v1",
        "required_decisions": [{"decision": "d", "owner": "compiler/codegen.py:absent"}],
        "existing_symbols": [{"surface_id": "s", "path": "compiler/codegen.py", "symbol": "absent"}],
    }
    with pytest.raises(ValueError, match="absent from initial source"):
        validate_against_package(load("t", "p", body=body), tmp_path)


# ---------------------------------------------------------------------------------------------
# THE MUTATION: a candidate that edits outside the contract is rejected. Run against a synthetic
# package shaped like the real one, so it runs everywhere rather than only where the artifacts are.
# ---------------------------------------------------------------------------------------------


def _synthetic(root: Path) -> None:
    (root / "mlir_oot/lowering").mkdir(parents=True)
    (root / "manifest.yaml").write_text("entrypoints: {tool: x}\noptimization_surfaces: []\n")
    (root / "mlir_oot/lowering/isa.py").write_text(
        "DIM = 16\n"
        "FUNCT = {'MVIN': 2}\n"
        "def _tile_word(addr, cols, rows):\n return addr\n"
        "def readout_plan(commit):\n return 'native'\n"
        "def build_trace(program):\n return []\n"
    )


def _contract() -> dict:
    return load(
        "t",
        "p",
        body={
            "schema": "compiler_edit_contract_v1",
            "required_decisions": [
                {"decision": "epilogue_placement", "owner": "mlir_oot/lowering/isa.py:readout_plan"}
            ],
            "existing_symbols": [
                {"surface_id": "epi", "path": "mlir_oot/lowering/isa.py", "symbol": "readout_plan"},
                {"surface_id": "grp", "path": "mlir_oot/lowering/isa.py", "symbol": "build_trace"},
            ],
            "helper_extensions": [{"directory": "mlir_oot/lowering", "surface_ids": ["epi"]}],
        },
    )


@pytest.mark.parametrize(
    "edit,allowed,why",
    [
        ("owned_symbol", True, "the decision under study must be reachable, or nothing is"),
        ("new_helper_module", True, "a scheduling helper may be added in the owning directory"),
        ("encoding_constant", False, "DIM is a hardware fact, not a scheduling choice"),
        ("encoding_table", False, "the funct table is the ISA"),
        ("address_helper", False, "_tile_word is ABI: it changes what every instruction means"),
        ("manifest_self_grant", False, "a candidate cannot widen its own authority"),
        ("unowned_new_file", False, "a new file outside a helper directory owns nothing"),
    ],
)
def test_candidate_editing_outside_the_contract_is_rejected(tmp_path, edit, allowed, why):
    before, after = tmp_path / "before", tmp_path / "after"
    _synthetic(before)
    _synthetic(after)
    contract = _contract()
    validate_against_package(contract, before)
    isa = after / "mlir_oot/lowering/isa.py"

    if edit == "owned_symbol":
        isa.write_text(isa.read_text().replace("return 'native'", "return 'scratch'"))
    elif edit == "new_helper_module":
        (after / "mlir_oot/lowering/tiling_helper.py").write_text("def pick():\n return 1\n")
    elif edit == "encoding_constant":
        isa.write_text(isa.read_text().replace("DIM = 16", "DIM = 32"))
    elif edit == "encoding_table":
        isa.write_text(isa.read_text().replace("{'MVIN': 2}", "{'MVIN': 3}"))
    elif edit == "address_helper":
        isa.write_text(isa.read_text().replace("return addr\n", "return addr + 1\n"))
    elif edit == "manifest_self_grant":
        (after / "manifest.yaml").write_text("entrypoints: {tool: x}\noptimization_surfaces: []\nextra: granted\n")
    else:
        (after / "mlir_oot/elsewhere.py").write_text("def new():\n return 1\n")

    report = inspect_compiler_edits(before, after, contract)
    assert (report["status"] == "allowed") is allowed, f"{edit}: {why} -- {report['violations']}"
    assert report["candidate_manifest_grants_authority"] is False
    assert report["contract_sha256"] == contract["sha256"]


def test_the_seal_binds_the_argument_not_just_the_symbols():
    """Changing the RECORDED REASON for a boundary changes the contract's identity.

    The `excluded` block is prose and `compiler_edit_scope` never reads it. It still travels inside
    the sealed body, so rewriting the argument for a boundary produces a different contract sha and
    shows up in every inspection report that cites one. An exclusion whose reason can be edited
    without trace is an exclusion nobody can audit.
    """
    body = {
        "schema": "compiler_edit_contract_v1",
        "required_decisions": [{"decision": "d", "owner": "a.py:f"}],
        "existing_symbols": [{"surface_id": "s", "path": "a.py", "symbol": "f"}],
        "excluded": [{"what": "b.py", "why": "original reason"}],
    }
    first = load("t", "p", body=body)
    second = load("t", "p", body={**body, "excluded": [{"what": "b.py", "why": "a different reason"}]})
    assert first["sha256"] != second["sha256"]


@pytest.mark.parametrize("target,package_id", DECLARED)
def test_declared_paths_are_scoped_and_relative(target, package_id):
    """No absolute path, no escape, and ancillary authority stays in docs/ or tests/."""
    body = _body(target, package_id)
    for entry in body["existing_symbols"]:
        path = entry["path"]
        assert not path.startswith("/") and ".." not in Path(path).parts, path
        assert path.endswith(".py"), path
    for ancillary in body.get("ancillary_paths", []):
        assert Path(ancillary).parts[0] in ("docs", "tests"), ancillary


@pytest.mark.parametrize("target,package_id", DECLARED)
def test_every_helper_surface_id_has_an_owning_symbol(target, package_id):
    """`validate_edit_contract` enforces this; asserting it here keeps the failure legible."""
    body = _body(target, package_id)
    owned = {entry["surface_id"] for entry in body["existing_symbols"]}
    for extension in body.get("helper_extensions", []):
        assert set(extension["surface_ids"]) <= owned, (
            f"{extension['directory']} names surface ids with no host-frozen owner: "
            f"{sorted(set(extension['surface_ids']) - owned)}"
        )


def test_seal_matches_the_digest_compiler_edit_scope_recomputes():
    body = {"schema": "compiler_edit_contract_v1", "existing_symbols": [{"a": 1}]}
    expected = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert seal(body)["sha256"] == expected
