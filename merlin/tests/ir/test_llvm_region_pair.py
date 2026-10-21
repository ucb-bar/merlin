"""Compressed syntax has two explicit graphs, never one semantic environment."""

import difflib
import json

import pytest
from test_llvm_region_boundary import module, source

from merlin.llvmlower.llvm_region_boundary import LLVMRegionBoundary


def correspondence(a, b):
    left = [(line, a.line_blocks.get(i)) for i, line in enumerate(a.lines)]
    right = [(line, b.line_blocks.get(i)) for i, line in enumerate(b.lines)]
    matches = difflib.SequenceMatcher(None, left, right, autojunk=False).get_matching_blocks()
    pairs = tuple((i + k, j + k) for i, j, n in matches for k in range(n))
    return dict(
        unchanged_lines=pairs,
        changed_original=set(range(len(a.lines))) - {i for i, _ in pairs},
        changed_candidate=set(range(len(b.lines))) - {j for _, j in pairs},
    )


def budget(a, b, **kw):
    ia = a.dependency_union(a.closed_region("header", "exit"))
    ib = b.dependency_union(b.closed_region("header", "exit"))
    return a.proof_pair_budget(b, ia, ib, **correspondence(a, b), **kw)


@pytest.mark.parametrize("bits,count", [(16, 3), (32, 19), (64, 7)])
def test_identical_bytes_have_two_explicit_memberships(bits, count):
    a, b = [module(source(bits, count), bits) for _ in range(2)]
    result = budget(a, b)
    witness = result["representation"]
    assert result["bytes"] == len(json.dumps(witness, sort_keys=True, separators=(",", ":")).encode())
    assert result["components"] == sum(result["physical_records"].values())
    assert result["original"] == result["candidate"]
    assert witness["candidate_cfg_delta"] == {} and witness["candidate_fact_delta"] == []
    assert witness["graph_local_semantics"].startswith("distinct")
    assert witness["selected_line_ranges"][0] == witness["selected_line_ranges"][1]


def test_changed_dependency_keeps_both_versions_despite_shared_use_bytes():
    a = module()
    b = module(source().replace("add i64 %row, 1", "add i64 %row, 2"))
    result = budget(a, b)
    witness = result["representation"]
    assert any(code == "%next = add i64 %row, 1" for _, code in witness["original_facts"])
    assert any(code == "%next = add i64 %row, 2" for _, code in witness["candidate_fact_delta"])
    assert witness["candidate_fact_references"]
    assert witness["graph_sources"][0] != witness["graph_sources"][1]
    # Accounting alone makes no assertion about this changed loop's semantics.


def test_new_block_and_rewired_edge_are_explicit_candidate_delta():
    a = module()
    text = source().replace(
        "  br label %header\nheader:",
        "  br label %bridge\nbridge:\n  %extra = add i64 1, 2\n  br label %header\nheader:",
        1,
    )
    text = text.replace("[ 0, %entry ]", "[ 0, %bridge ]")
    b = module(text)
    result = budget(a, b)
    delta = result["representation"]["candidate_cfg_delta"]
    assert set(delta) == {"entry", "bridge"}
    assert delta["entry"] == (("bridge", None, True),)
    assert result["candidate"]["blocks"] == result["original"]["blocks"] + 1
    assert result["candidate"]["edges"] == result["original"]["edges"] + 1
    assert any("%extra =" in code for _, code in result["representation"]["candidate_fact_delta"])


@pytest.mark.parametrize("kind", ["duplicate", "reverse", "outside", "bool", "uncovered", "wrong_bytes", "wrong_block"])
def test_bad_or_incomplete_correspondence_refuses(kind):
    a, b = module(), module()
    args = correspondence(a, b)
    pairs = list(args["unchanged_lines"])
    if kind == "duplicate":
        pairs.insert(1, pairs[0])
    elif kind == "reverse":
        pairs.reverse()
    elif kind == "outside":
        pairs[-1] = (len(a.lines), len(b.lines))
    elif kind == "bool":
        pairs[0] = (False, 0)
    elif kind == "uncovered":
        pairs.pop()
    elif kind == "wrong_bytes":
        b = module(source().replace("i64 %row)", "i64 1)", 1))
    else:
        i = next(i for i in a.blocks["entry"] if "call void" in a.lines[i])
        j = next(i for i in b.blocks["pad"] if "call void" in b.lines[i])
        assert a.lines[i] == b.lines[j]
        pairs = [(i, j)]
        args["changed_original"] = set(range(len(a.lines))) - {i}
        args["changed_candidate"] = set(range(len(b.lines))) - {j}
    args["unchanged_lines"] = tuple(pairs)
    with pytest.raises(ValueError):
        a.proof_pair_budget(b, set(), set(), **args)


@pytest.mark.parametrize(
    "kind", ["index_bool", "negative", "iterator", "changed_iterator", "changed_outside", "map_list"]
)
def test_types_and_scan_membership_refuse_before_representation(kind):
    a, b = module(), module()
    args = correspondence(a, b)
    indices = set()
    if kind == "index_bool":
        indices = {True}
    elif kind == "negative":
        indices = {-1}
    elif kind == "iterator":
        indices = iter([0])
    elif kind == "changed_iterator":
        args["changed_original"] = iter([0])
    elif kind == "changed_outside":
        args["changed_candidate"] = {len(b.lines)}
    else:
        args["unchanged_lines"] = list(args["unchanged_lines"])
    with pytest.raises(ValueError):
        a.proof_pair_budget(b, indices, set(), **args)


def test_changed_code_is_charged_even_when_caller_omits_it_from_union():
    a = module()
    b = module(source().replace("add i64 %row, 1", "add i64 %row, 3"))
    result = a.proof_pair_budget(b, set(), set(), **correspondence(a, b))
    assert result["original"]["facts"] == result["candidate"]["facts"] == 1
    assert "add i64 %row, 3" in result["representation"]["candidate_fact_delta"][0][1]


def test_stale_actual_index_and_scan_limit_drift_refuse():
    a, b = module(), module()
    args = correspondence(a, b)
    b.edges["entry"] = (("pad", None, True),)
    with pytest.raises(ValueError, match="actual source"):
        a.proof_pair_budget(b, set(), set(), **args)
    b = module(max_cfg_blocks=100)
    with pytest.raises(ValueError, match="scan limits"):
        a.proof_pair_budget(b, set(), set(), **correspondence(a, b))


def test_actual_serialized_bytes_and_physical_components_keep_original_caps():
    a, b = module(), module()
    result = budget(a, b)
    with pytest.raises(ValueError, match="representation bound"):
        budget(a, b, max_bytes=result["bytes"] - 1)
    with pytest.raises(ValueError, match="representation bound"):
        budget(a, b, max_components=result["components"] - 1)
    for kw in ({"max_bytes": 65537}, {"max_components": 4097}, {"max_bytes": True}):
        with pytest.raises(ValueError, match="bounded"):
            budget(a, b, **kw)


def long_source(count=750):
    names = [f"block_{i:04}_with_context" for i in range(count)]
    body = ["define void @walk() {\n"]
    for i, name in enumerate(names):
        body.append(name + ":\n")
        body.append(f"  br label %{names[i + 1]}\n" if i + 1 < len(names) else "  ret void\n")
    return "".join(body) + "}\n"


def test_compressed_pair_reconstructs_two_large_cfgs_without_double_charge():
    text = long_source()
    a, b = [LLVMRegionBoundary(text, function_name="walk", argument_types=()) for _ in range(2)]
    one = a.proof_budget({a.first})
    assert one["bytes"] * 2 > 65536
    result = a.proof_pair_budget(b, {a.first}, {b.first}, **correspondence(a, b))
    assert result["bytes"] <= 65536 and result["components"] <= 4096
    assert result["original"]["blocks"] == result["candidate"]["blocks"] == 750
    # A claimed copied graph with unrelated renamed blocks cannot receive refs.
    c = LLVMRegionBoundary(text.replace("block_", "other_"), function_name="walk", argument_types=())
    with pytest.raises(ValueError, match="representation bound"):
        a.proof_pair_budget(c, {a.first}, {c.first}, **correspondence(a, c))


def test_existing_subclass_path_uses_actual_reparsed_code_not_overrides():
    class SelectedModule(LLVMRegionBoundary):
        def code(self, index):
            raise AssertionError("must use the independent actual base parser")

    a = SelectedModule(source(), function_name="work", argument_types=("ptr", "i64"))
    b = SelectedModule(source(), function_name="work", argument_types=("ptr", "i64"))
    result = a.proof_pair_budget(b, {a.first}, {b.first}, **correspondence(a, b))
    assert result["original"]["facts"] == 1


def test_mutated_token_span_cannot_invent_or_omit_charged_fact():
    a, b = module(), module()
    args = correspondence(a, b)
    i = b.definition_lines["%next"]
    b.tokens[i] = list(b.tokens[b.first])
    with pytest.raises(ValueError, match="actual source"):
        a.proof_pair_budget(b, {a.first}, {b.first, i}, **args)


@pytest.mark.parametrize("forge", [False, True])
def test_actual_argument_namespace_mismatch_cannot_be_forged(forge):
    a = module()
    b = module(source().replace("%limit", "%foreign"))
    args = correspondence(a, b)
    if forge:
        b.arguments = a.arguments
    with pytest.raises(ValueError, match="namespace|actual source"):
        a.proof_pair_budget(b, set(), set(), **args)
