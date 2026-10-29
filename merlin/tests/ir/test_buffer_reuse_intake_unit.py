"""SDK-free bounded buffer candidate tests; no Lean acceptance or allocator approval."""

import copy
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.verify.buffer_reuse_intake import ALLOCATOR, ROOT, prepare, revalidate


@pytest.fixture(scope="module")
def raw_case():
    default = repo_root().parent / "merlin/out/artifacts/verification/graph-shape-normalization"
    base = Path(os.environ.get("MERLIN_BUFFER_REUSE_CASE_ROOT", default))
    paths = [
        base / "epoch-cases-v1/two_epochs/source.mlir",
        base / "epoch-cases-v1/two_epochs/memory/after.mlir",
        base / "after-binding-deadline-v1/tests-v9/pytest/test_actual_middle_is_checked_0/PREPARATION.json",
    ]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        pytest.skip("explicit retained buffer input data unavailable: " + str(missing))
    return paths[0].read_text(), paths[1].read_text(), json.loads(paths[2].read_text())


@pytest.fixture(scope="module")
def candidate(raw_case):
    return prepare(*raw_case, allocator=ALLOCATOR)


def test_complete_candidate_is_explicitly_non_authoritative(candidate):
    report = candidate.describe()
    assert report["final_root"] == ROOT
    assert ROOT == candidate.roots[-1] and len(set(candidate.roots)) == len(candidate.roots)
    assert report["relation"] == "equal_possible_successful_public_observations"
    assert not report["authority"] and not report["pipeline_receipt_issued"]
    assert not report["full_error_or_world_equivalence"]
    assert revalidate(candidate) == {"fresh": True, "authority": False, "kernel_outcome": "UNOBSERVED"}


@pytest.mark.parametrize("allocator", ["", "finite_capacity", "observable_pointers"])
def test_allocator_contract_refusals(raw_case, allocator):
    with pytest.raises(ValueError, match="8->16-byte"):
        prepare(*raw_case, allocator=allocator)


@pytest.mark.parametrize("bits", [32, True])
def test_domain_width_refusals(raw_case, bits):
    with pytest.raises(ValueError, match="index64"):
        prepare(*raw_case, allocator=ALLOCATOR, index_bits=bits)


@pytest.mark.parametrize("field", ["query", "roots", "domain", "entry_roster_json"])
def test_changed_saved_candidate_cannot_revalidate(candidate, field):
    altered = {
        "query": candidate.query + "\n",
        "roots": (ROOT,),
        "domain": "finite-allocator",
        "entry_roster_json": "[]",
    }
    with pytest.raises(ValueError, match="stale/forged"):
        revalidate(replace(candidate, **{field: altered[field]}))


def test_fake_hint_approval_is_not_authority(raw_case, candidate):
    source, final, hint = raw_case
    fake = copy.deepcopy(hint)
    fake.update(kernel_checked=True, qualified=True, human_approved=True)
    result = prepare(source, final, fake, allocator=ALLOCATOR)
    assert result.query == candidate.query
    assert result.describe()["authority"] is False


def test_raw_chain_reconstructs_without_saved_plan(raw_case):
    from merlin.verify.buffer_reuse_intake import prepare_chain

    default = repo_root().parent / "merlin/out/artifacts/verification/graph-shape-normalization"
    base = Path(os.environ.get("MERLIN_BUFFER_REUSE_CASE_ROOT", default)) / "epoch-cases-v1/two_epochs"
    buffers = tuple(
        (base / name).read_text()
        for name in ("source.mlir", "shape/after.mlir", "memory/middle.mlir", "memory/after.mlir")
    )
    candidate = prepare_chain(buffers, allocator=ALLOCATOR)
    assert candidate.source == buffers[0] and candidate.final == buffers[-1]
    assert len(candidate.describe()["raw_chain_sha256"]) == 4
    assert revalidate(candidate)["authority"] is False


@pytest.mark.parametrize("mutation", ["missing", "swapped_middle", "changed_control", "stale_chain"])
def test_complete_raw_chain_cannot_be_substituted(raw_case, mutation):
    from merlin.verify.buffer_reuse_intake import prepare_chain

    default = repo_root().parent / "merlin/out/artifacts/verification/graph-shape-normalization"
    base = Path(os.environ.get("MERLIN_BUFFER_REUSE_CASE_ROOT", default)) / "epoch-cases-v1/two_epochs"
    buffers = tuple(
        (base / name).read_text()
        for name in ("source.mlir", "shape/after.mlir", "memory/middle.mlir", "memory/after.mlir")
    )
    candidate = prepare_chain(buffers, allocator=ALLOCATOR)
    if mutation == "stale_chain":
        changed = replace(candidate, raw_chain_json=json.dumps((buffers[0], buffers[2], buffers[1], buffers[3])))
        with pytest.raises(ValueError):
            revalidate(changed)
        return
    if mutation == "missing":
        changed = buffers[:-1]
    elif mutation == "swapped_middle":
        changed = (buffers[0], buffers[2], buffers[1], buffers[3])
    else:
        changed = (buffers[0], buffers[1], buffers[2], buffers[3].replace("2 : index", "1 : index"))
        assert changed[-1] != buffers[-1]
    try:
        result = prepare_chain(changed, allocator=ALLOCATOR)
    except ValueError:
        return  # host cannot prepare the requested exact finite obligations
    assert result.query != candidate.query
    assert result.describe()["authority"] is False  # no inferred Lean refusal


def test_unmodeled_external_declaration_is_explicitly_outside_entry_domain(raw_case):
    source, final, hint = raw_case
    import io

    from xdsl.dialects import func
    from xdsl.printer import Printer

    from merlin.verify.buffer_reuse_bridge import parse

    changed = []
    for text in (source, final):
        module = parse(text)
        module.body.block.add_op(func.FuncOp.external("external", [], []))
        module.verify()
        stream = io.StringIO()
        Printer(stream=stream, print_generic_format=True).print_op(module)
        changed.append(stream.getvalue())
    with pytest.raises(ValueError):
        prepare(*changed, hint, allocator=ALLOCATOR)
