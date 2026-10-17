"""End-to-end composition (design C10): recorded preprocessing boundaries of a real Merlin pass run,
each checked, plus the capture's support, compose into verified refinement evidence. Overall
correctness remains abstracted without the separate source-definedness proof."""

from __future__ import annotations

import copy
import json
import sys
from functools import partial
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.llvmlower.passes_xdsl import preprocess_text
from merlin.verify.lean import stages as lean_stages

FIXTURE = repo_root() / "merlin" / "tests" / "fixtures" / "lean_dce_forward.mlir"
SCHEMA = repo_root() / "merlin" / "contract" / "schemas" / "lean_chain.schema.json"


@pytest.mark.parametrize("change", ["unchanged", "modified", "deleted", "wrong_size"])
def test_chain_revalidates_handoff_file(tmp_path, monkeypatch, change):
    from merlin.verify.lean import check

    # Exercise composition without external tools: these stand in only for the identities used by
    # _stale, while the recorded stage, support and handoff files are real files.
    checker = tmp_path / "checker"
    checker.write_text("checker")
    (tmp_path / "SPEC_LOCK").write_text("lock")
    monkeypatch.setattr(check, "lean_exe", lambda: checker)
    monkeypatch.setattr(check, "lean_root", lambda: tmp_path)
    monkeypatch.setattr(check, "_replay", lambda *args, **kwargs: [])
    lean = {
        "checker_sha256": check._sha(checker),
        "spec_lock_sha256": check._sha(tmp_path / "SPEC_LOCK"),
    }
    stage = tmp_path / "stage"
    stage.mkdir()
    before, after = stage / "before.mlir", stage / "after.mlir"
    before.write_text(FIXTURE.read_text())
    after.write_bytes(before.read_bytes())
    (stage / "receipt.json").write_text(
        json.dumps(
            {
                "claim_type": "refine_conditional",
                "boundary": "identity",
                "verdict": "verified",
                "before": check._file_ref(before),
                "after": check._file_ref(after),
                "lean": lean,
            }
        )
    )
    support = tmp_path / "support.json"
    support.write_text(
        json.dumps(
            {
                "claim_type": "not_stuck",
                "boundary": "support",
                "function": "forward",
                "verdict": "verified",
                "subject": check._file_ref(before),
                "domain": "typed inputs",
                "index_bits": 64,
                "lean": lean,
            }
        )
    )
    handed = tmp_path / "model.upstream.mlir"
    handed.write_bytes(after.read_bytes())
    monkeypatch.setenv("MERLIN_LEAN_STAGES", str(tmp_path))
    lean_stages.record_handoff(handed)
    handoff = tmp_path / "handoff.json"
    if change == "modified":
        handed.write_text(handed.read_text().replace("arith.muli", "arith.addi"))
        assert handed.stat().st_size == after.stat().st_size
        assert handed.read_bytes() != after.read_bytes()
    elif change == "deleted":
        handed.unlink()
    elif change == "wrong_size":
        ref = json.loads(handoff.read_text())
        ref["bytes"] += 1
        handoff.write_text(json.dumps(ref))
    doc = check.compose_chain(
        [stage], support, "forward", tmp_path / "chain.json", handoff=handoff, m2m_python=Path(sys.executable)
    )
    assert doc["refinement_verdict"] == ("verified" if change == "unchanged" else "abstracted")
    assert doc["verdict"] == "abstracted"
    if change != "unchanged":
        assert "handoff" in doc["reason"]


def test_chain_composes_real_preprocessing(tmp_path, monkeypatch, lean_tools):
    import jsonschema

    from merlin.verify.lean.check import check_stage, check_support, compose_chain

    compose_chain = partial(compose_chain, m2m_python=lean_tools.m2m_python)
    monkeypatch.setenv("MERLIN_LEAN_STAGES", str(tmp_path / "stages"))
    preprocess_text(FIXTURE.read_text())
    stages = lean_stages.completed(tmp_path / "stages")
    for stage in stages:
        assert check_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    capture = stages[0] / "before.mlir"
    assert capture.read_text() == FIXTURE.read_text()
    check_support(capture, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)
    schema = json.loads(SCHEMA.read_text())

    doc = compose_chain(
        stages,
        tmp_path / "support" / "receipt.json",
        "forward",
        tmp_path / "chain.json",
        m2m_python=lean_tools.m2m_python,
    )
    jsonschema.validate(doc, schema)
    assert doc["chained"] and doc["refinement_verdict"] == "verified", doc.get("reason")
    assert doc["verdict"] == "abstracted"
    assert doc["source_definedness"]["verdict"] == "unmeasured"
    assert [link["boundary"] for link in doc["links"]] == [
        "xdsl-parse",
        "xdsl-pruned",
        "xdsl-quant-lowered",
        "xdsl-c-interface",
    ]

    # Dropping a boundary breaks the chain: the composition must not verify.
    broken = compose_chain(
        [stages[0], *stages[2:]], tmp_path / "support" / "receipt.json", "forward", tmp_path / "broken.json"
    )
    jsonschema.validate(broken, schema)
    assert not broken["chained"] and broken["refinement_verdict"] == "abstracted"
    assert broken["verdict"] == "abstracted"

    # The chain must end at the bytes handed to the lowering.
    support = tmp_path / "support" / "receipt.json"
    handed = tmp_path / "handed.mlir"
    handed.write_bytes((stages[-1] / "after.mlir").read_bytes())
    lean_stages.record_handoff(handed)
    handoff = tmp_path / "stages" / "handoff.json"
    ok = compose_chain(
        stages, support, "forward", tmp_path / "handoff-ok.json", handoff=handoff, m2m_python=lean_tools.m2m_python
    )
    jsonschema.validate(ok, schema)
    assert ok["refinement_verdict"] == "verified"
    assert ok["verdict"] == "abstracted"
    assert ok["handoff_sha256"] == json.loads(handoff.read_text())["sha256"]
    handed.write_text(handed.read_text() + "\n// edited after preprocessing\n")
    lean_stages.record_handoff(handed)
    other = compose_chain(stages, support, "forward", tmp_path / "handoff-bad.json", handoff=handoff)
    assert not other["chained"] and other["refinement_verdict"] == "abstracted"
    assert other["verdict"] == "abstracted"

    # A receipt whose stage file changed after it was checked is stale.
    after = stages[1] / "after.mlir"
    after.write_text(after.read_text() + "\n")
    stale = compose_chain(stages, support, "forward", tmp_path / "stale.json")
    jsonschema.validate(stale, schema)
    assert stale["verdict"] == "abstracted" and "changed or missing" in stale["reason"]
    assert stale["refinement_verdict"] == "abstracted"


def test_defined_chain_replays_real_preprocessing(tmp_path, monkeypatch, lean_tools):
    import jsonschema

    from merlin.verify.lean.check import check_definedness, check_stage, check_support, compose_chain

    monkeypatch.setenv("MERLIN_LEAN_STAGES", str(tmp_path / "stages"))
    source = """func.func @forward(%a: i32, %b: i32) -> i32 {
      %unused = arith.muli %a, %b : i32
      %r = arith.addi %a, %b : i32
      return %r : i32
    }"""
    preprocess_text(source)
    stages = lean_stages.completed(tmp_path / "stages")
    for stage in stages:
        assert check_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    capture = stages[0] / "before.mlir"
    support = tmp_path / "support"
    defined = tmp_path / "defined"
    assert check_support(capture, "forward", support, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    receipt = check_definedness(capture, "forward", defined, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "verified", receipt.get("reason")

    def compose(path=defined / "receipt.json"):
        return compose_chain(stages, support / "receipt.json", "forward", tmp_path / "chain.json",
                             m2m_python=lean_tools.m2m_python, definedness=path)

    result = compose()
    jsonschema.validate(result, json.loads(SCHEMA.read_text()))
    assert result["verdict"] == result["refinement_verdict"] == "verified", result.get("reason")
    assert result["source_definedness"]["verdict"] == "verified"
    assert result["theorem"] == "MerlinVerify.Check.chain_defined_refines"
    assert result["domain"] == receipt["domain"] and result["domain"].startswith("nonpoison inputs")
    assert "reason" not in result
    missing = compose(None)
    assert missing["verdict"] == "abstracted" and missing["refinement_verdict"] == "verified"
    assert missing["source_definedness"]["verdict"] == "unmeasured"

    # Support cannot be substituted for definedness, even for the same function and bytes.
    wrong_obligation = compose(support / "receipt.json")
    assert wrong_obligation["verdict"] == "abstracted"
    assert "required proof obligation" in wrong_obligation["reason"]

    for keys, value in [
        (("domain",), "all inputs"), (("function",), "other"), (("index_bits",), 32),
        (("axioms",), ["sorryAx"]), (("assumptions",), []),
        (("exporter", "round_trip"), "not_run"), (("theorem",), "unrelated.theorem"),
    ]:
        altered = copy.deepcopy(receipt)
        node = altered
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
        (defined / "receipt.json").write_text(json.dumps(altered))
        assert compose()["verdict"] != "verified", keys
    (defined / "receipt.json").write_text(json.dumps(receipt))

    # A valid proof about another source is still irrelevant to this chain.
    other = tmp_path / "other.mlir"
    other.write_text("func.func @forward(%a: i32, %b: i32) -> i32 { return %a : i32 }")
    assert check_definedness(other, "forward", tmp_path / "other-proof",
                             m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    unrelated = compose(tmp_path / "other-proof/receipt.json")
    assert unrelated["verdict"] == "abstracted" and "another source" in unrelated["reason"]
    assert check_definedness(capture, "forward", tmp_path / "width32",
                             m2m_python=lean_tools.m2m_python, index_bits=32)["verdict"] == "verified"
    another_width = compose(tmp_path / "width32/receipt.json")
    assert another_width["verdict"] == "abstracted" and "another source or index width" in another_width["reason"]


@pytest.mark.parametrize("artifact", ["ir", "receipt", "handoff"])
def test_chain_rechecks_earlier_files_before_publication(tmp_path, monkeypatch, lean_tools, artifact):
    from merlin.verify.lean import check

    stage = tmp_path / "stage"
    stage.mkdir()
    source = "func.func @forward(%a: i32) -> i32 { return %a : i32 }"
    for name in ("before", "after"):
        (stage / f"{name}.mlir").write_text(source)
    (stage / "meta.json").write_text('{"boundary":"identity"}')
    assert check.check_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    support, defined = tmp_path / "support", tmp_path / "defined"
    assert check.check_support(stage / "before.mlir", "forward", support,
                               m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    assert check.check_definedness(stage / "before.mlir", "forward", defined,
                                   m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    monkeypatch.setenv("MERLIN_LEAN_STAGES", str(tmp_path / "recorded"))
    handed = tmp_path / "handed.mlir"
    handed.write_text(source)
    lean_stages.record_handoff(handed)
    handoff = tmp_path / "recorded/handoff.json"
    validate = check._validate_receipt

    def edit_after_replay(receipt, **kwargs):
        result = validate(receipt, **kwargs)
        if kwargs["obligation"] == check._DEFINED:
            if artifact == "ir":
                (stage / "after.mlir").write_text(source + "\n// changed during a later replay\n")
            elif artifact == "receipt":
                path = stage / "receipt.json"
                path.write_text(path.read_text() + "\n")
            else:
                handed.write_text(source + "\n// changed during a later replay\n")
        return result

    monkeypatch.setattr(check, "_validate_receipt", edit_after_replay)
    result = check.compose_chain([stage], support / "receipt.json", "forward", tmp_path / "chain.json",
                                 m2m_python=lean_tools.m2m_python, definedness=defined / "receipt.json",
                                 handoff=handoff)
    assert result["verdict"] == result["refinement_verdict"] == "abstracted"
    assert "changed" in result["reason"]


def test_dense_tensor_preprocessing_chain_is_defined(tmp_path, monkeypatch, lean_tools):
    from merlin.verify.lean.check import check_definedness, check_stage, check_support, compose_chain

    monkeypatch.setenv("MERLIN_LEAN_STAGES", str(tmp_path / "stages"))
    preprocess_text("""func.func @forward() -> tensor<6xi32> {
      %unused = tensor.empty() : tensor<4xi32>
      %weights = arith.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
      %r = tensor.collapse_shape %weights [[0, 1]] : tensor<2x3xi32> into tensor<6xi32>
      return %r : tensor<6xi32>
    }""")
    stages = lean_stages.completed(tmp_path / "stages")
    for stage in stages:
        assert check_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    capture = stages[0] / "before.mlir"
    assert check_support(capture, "forward", tmp_path / "support",
                         m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    assert check_definedness(capture, "forward", tmp_path / "defined",
                             m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    result = compose_chain(stages, tmp_path / "support/receipt.json", "forward", tmp_path / "chain.json",
                           m2m_python=lean_tools.m2m_python, definedness=tmp_path / "defined/receipt.json")
    assert result["verdict"] == "verified", result.get("reason")
