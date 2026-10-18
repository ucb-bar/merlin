"""Legacy and extended totality claims replay with their actual theorem identities."""

import json

import jsonschema
import pytest

from merlin.common.paths import repo_root
from merlin.llvmlower.passes_xdsl import preprocess_text
from merlin.verify.lean import check, stages

SOURCE = """func.func @forward(%a: i13, %b: i13) -> i13 {
  %dead = arith.muli %a, %b : i13
  %r = arith.addi %a, %b : i13
  return %r : i13
}"""


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_definedness_identity_and_chain_replay(tmp_path, monkeypatch, lean_tools, legacy, index_bits):
    monkeypatch.setenv(stages.ENV, str(tmp_path / "stages"))
    preprocess_text(SOURCE)
    recorded = stages.completed(tmp_path / "stages")
    source = recorded[0] / "before.mlir"
    for stage in recorded:
        assert (
            check.check_typed_stage(stage, m2m_python=lean_tools.m2m_python, index_bits=index_bits)["verdict"]
            == "verified"
        )
    assert (
        check.check_support(
            source, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python, index_bits=index_bits
        )["verdict"]
        == "verified"
    )
    if legacy:
        # Exercise the retained old executable obligation, not a renamed modern receipt.
        receipt = check._check_function(
            check._DEFINED,
            source,
            "forward",
            tmp_path / "defined",
            m2m_python=lean_tools.m2m_python,
            index_bits=index_bits,
        )
        checker, theorem = check.DEFINED_CHECKER, check.DEFINED_THEOREM
    else:
        receipt = check.check_definedness(
            source, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python, index_bits=index_bits
        )
        checker, theorem = check.EXTENDED_DEFINED_CHECKER, check.EXTENDED_DEFINED_THEOREM
    assert receipt["verdict"] == "verified", receipt.get("reason")
    assert (receipt["checker"], receipt["theorem"]) == (checker, theorem)
    chain = check.compose_chain(
        recorded,
        tmp_path / "support/receipt.json",
        "forward",
        tmp_path / "chain.json",
        definedness=tmp_path / "defined/receipt.json",
        m2m_python=lean_tools.m2m_python,
    )
    assert chain["verdict"] == "verified", chain.get("reason")
    assert (chain["source_definedness"]["checker"], chain["source_definedness"]["theorem"]) == (checker, theorem)
    schema = json.loads((repo_root() / "merlin/contract/schemas/lean_chain.schema.json").read_text())
    jsonschema.validate(chain, schema)
    # A valid-looking receipt cannot substitute a theorem belonging to another checker.
    receipt["theorem"] = check.EXTENDED_DEFINED_THEOREM if legacy else check.DEFINED_THEOREM
    (tmp_path / "defined/receipt.json").write_text(json.dumps(receipt))
    refused = check.compose_chain(
        recorded,
        tmp_path / "support/receipt.json",
        "forward",
        tmp_path / "refused.json",
        definedness=tmp_path / "defined/receipt.json",
        m2m_python=lean_tools.m2m_python,
    )
    assert refused["verdict"] == "abstracted"
    assert "required proof obligation" in refused["reason"]
    chain["source_definedness"]["theorem"] = receipt["theorem"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(chain, schema)
