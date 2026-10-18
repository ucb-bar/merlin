"""Long source reshape chains use checked transitive runtime provenance."""

import subprocess

import pytest
from test_lean_reshape_compose import _canonicalize, _type
from test_lean_reshape_mixed import MUTATE

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage, compose_chain


def _source(cycles=2, shape=(2, 3, 2, 2), element="i32", *, escape=False):
    a, b, c, d = shape
    original, middle, expanded, final = [
        _type(s, element) for s in [shape, (a * b, c * d), (a, b, c * d), (a * b * c * d,)]
    ]
    operations = [f"%s0 = tensor.collapse_shape %a [[0,1],[2,3]] : {original} into {middle}"]
    ids, types = ["%s0"], [middle]
    for n in range(cycles):
        operations += [
            f"%e{n} = tensor.expand_shape {ids[-1]} [[0,1],[2]] output_shape [{a},{b},{c * d}] "
            f": {middle} into {expanded}",
            f"%s{n + 1} = tensor.collapse_shape %e{n} [[0,1],[2]] : {expanded} into {middle}",
        ]
        ids += [f"%e{n}", f"%s{n + 1}"]
        types += [expanded, middle]
    operations += [f"%out = tensor.collapse_shape {ids[-1]} [[0,1]] : {middle} into {final}"]
    results, result_types = ["%out", "%a"], [final, original]
    if escape:
        results += ids
        result_types += types
    return f"""func.func @forward(%a: {original}, %alternate: {original}, %other_result: {final})
      -> ({", ".join(result_types)}) {{
      {" ".join(operations)}
      return {", ".join(results)} : {", ".join(result_types)}
    }}"""


@pytest.mark.parametrize("cycles", [1, 3, 12])
@pytest.mark.parametrize("escape", [False, True])
def test_actual_transitive_reshape_chain(tmp_path, lean_tools, cycles, escape):
    stage, before, after = _canonicalize(tmp_path, lean_tools.m2m_python, _source(cycles, escape=escape))
    if not escape:
        assert before.read_text().count("tensor.") > after.read_text().count("tensor.")
    assert "tensor.expand_shape" not in after.read_text()
    checked = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "verified", checked.get("reason")


@pytest.mark.parametrize("shape,element", [((0, 3, 2, 2), "i8"), ((2, 1, 2, 2), "f32"), ((2, 3, 2, 2), "bf16")])
def test_transitive_empty_unit_and_float_shapes(tmp_path, lean_tools, shape, element):
    stage, _, _ = _canonicalize(tmp_path, lean_tools.m2m_python, _source(3, shape, element))
    checked = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "verified", checked.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("escape", [False, True])
def test_transitive_strict_chain(tmp_path, lean_tools, index_bits, escape):
    stage, before, after = _canonicalize(tmp_path, lean_tools.m2m_python, _source(4, escape=escape))
    checked = check_typed_stage(stage, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "verified", checked.get("reason")
    for checker, name in [(check_support, "support"), (check_definedness, "defined")]:
        checked = checker(before, "forward", tmp_path / name, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
        assert checked["verdict"] == "verified", checked.get("reason")

    def replay():
        return compose_chain(
            [stage],
            tmp_path / "support/receipt.json",
            "forward",
            tmp_path / "chain.json",
            definedness=tmp_path / "defined/receipt.json",
            m2m_python=lean_tools.m2m_python,
        )

    checked = replay()
    assert checked["verdict"] == "verified", checked.get("reason")
    assert checked["theorem"] == "MerlinVerify.Check.chain_typed_defined_refines"
    after.write_text(after.read_text() + "\n// changed after receipt\n")
    assert replay()["verdict"] == "abstracted"


@pytest.mark.parametrize("fault", ["operand", "result", "escaping_input", "attribute"])
def test_transitive_verified_mlir_mutations(tmp_path, lean_tools, fault):
    stage, _, after = _canonicalize(tmp_path, lean_tools.m2m_python, _source(3))
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    changed = subprocess.run(
        [str(lean_tools.m2m_python), "-c", MUTATE, str(after), fault], capture_output=True, text=True, check=False
    )
    assert changed.returncode == 0, changed.stderr
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "abstracted"


def test_128_source_reshape_operations(tmp_path, lean_tools):
    stage, before, after = _canonicalize(tmp_path, lean_tools.m2m_python, _source(63))
    assert before.read_text().count("tensor.") == 128
    assert after.read_text().count("tensor.") == 1
    checked = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "verified", checked.get("reason")
