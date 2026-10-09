"""Actual public tools check local bit-vector facts and required ordinary gaps."""

import copy
import dataclasses
import importlib.util
import json
import os
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase0 import arithmetic_intake as R
from merlin_experiments.phase0 import component_arithmetic_obligations as O
from merlin_experiments.phase0 import component_automatic as A
from merlin_experiments.phase0.rtl_intake import RtlIntakeRefusal, issue_independent_hardware_intake

from merlin.common import invocation_record as I
from merlin.targetgen import golden_store
from merlin.targetgen.capsule_inputs import materialize_capsule_leaves
from merlin.targetgen.rtl.hw_arithmetic import local_arithmetic
from merlin.targetgen.rtl.hw_graph import parse_generic_hw
from merlin.targetgen.rtl.source_selection import produce_selection

_fixture_path = Path(__file__).with_name("test_component_automatic.py")
_fixture_spec = importlib.util.spec_from_file_location("private_arithmetic_source_fixtures", _fixture_path)
automatic_fixtures = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(automatic_fixtures)
automatic, independent = automatic_fixtures.automatic, automatic_fixtures.independent
write = automatic_fixtures.write
_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}


@pytest.fixture(scope="module")
def tools():
    names = ("MERLIN_TEST_FIRTOOL", "MERLIN_TEST_CIRCT_OPT", "MERLIN_TEST_VERILATOR")
    if any(not os.environ.get(name) for name in names):
        pytest.skip("local HW controls need explicit protected public native FIRRTL/CIRCT/Verilator tools")
    return {name: Path(os.environ[name]).absolute() for name in names}


@pytest.fixture
def selected(tools, tmp_path):
    source = tmp_path / "original.fir"
    source.write_text(
        "FIRRTL version 3.2.0\ncircuit Unit :\n"
        "  module Unit : @[generators/test_unit/src/Numeric.scala 1:1]\n"
        "    input x : SInt<8>\n    input y : SInt<8>\n"
        "    input z : SInt<32>\n    output q : SInt<20>\n"
        "    node product = mul(x, y)\n    q <= add(product, z)\n"
    )
    bundle = produce_selection(
        target="test_unit",
        firrtl=source,
        generator="test_unit",
        config="IndependentNumeric",
        core_root="Unit",
        firtool=tools["MERLIN_TEST_FIRTOOL"],
        output=tmp_path / "original-production",
    )
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("target: test_unit\n")
    private = tmp_path / "protected-answer"
    private.mkdir()
    return {
        "target": "test_unit",
        "descriptor": descriptor,
        "source_bundle": bundle,
        "forbidden_roots": (private,),
        "output": tmp_path / "issued-hardware",
    }


def generic(source, tools, root):
    root.mkdir(parents=True, exist_ok=False)
    original, observed = root / "original.hw.mlir", root / "generic.hw.mlir"
    original.write_text(source)
    I.run(
        [str(tools["MERLIN_TEST_CIRCT_OPT"]), "--mlir-print-op-generic", str(original), "-o", str(observed)],
        directory=root,
        stage="native_arithmetic_control_genericization",
        inputs=(original,),
        outputs=(observed,),
        env=_ENV,
        capture_output=True,
        check=True,
        timeout=60,
    )
    return local_arithmetic(parse_generic_hw(observed.read_text()))


def core(selected):
    record = json.loads(selected["source_bundle"].read_bytes())
    return Path(record["sources"]["core_hw"]["path"]).read_text()


def test_real_native_source_relation_and_live_issuance(selected, tools, tmp_path):
    hardware = issue_independent_hardware_intake(**selected)
    intake = R.issue_independent_arithmetic_intake(
        hardware=hardware,
        circt_opt=tools["MERLIN_TEST_CIRCT_OPT"],
        forbidden_roots=selected["forbidden_roots"],
        output=tmp_path / "arithmetic",
    )
    record = intake.record()
    assert record["facts"]["relations"] == [
        {
            "module": "Unit",
            "output_ordinal": 0,
            "law": "signed_multiply_add_modulo_result_width",
            "operands": [{"input_ordinal": 0, "width": 8}, {"input_ordinal": 1, "width": 8}],
            "addend": {"input_ordinal": 2, "width": 32},
            "product_bits": 16,
            "result_bits": 20,
        }
    ]
    assert record["facts"]["complete_arithmetic_domain"] is False
    R.verify_record(record)
    with pytest.raises(RtlIntakeRefusal, match="live independently issued"):
        dataclasses.replace(intake).verify()
    changed = copy.deepcopy(record)
    changed["facts"]["relations"][0]["result_bits"] = 32
    with pytest.raises(RtlIntakeRefusal, match="original typed SSA"):
        R.verify_record(changed)
    Path(next(pin.path for pin in intake.source_pins if pin.role == "generic-core-hw")).write_text("changed")
    with pytest.raises(RtlIntakeRefusal, match="source changed"):
        intake.verify()


@pytest.mark.parametrize(
    "mutation",
    ["wrong_sign", "wrong_product_sign", "shifted_addend", "subtract", "nary_add", "zero_extension", "state"],
)
def test_actual_native_structural_defects_cannot_mint_signed_relation(selected, tools, tmp_path, mutation):
    source = core(selected)
    if mutation == "wrong_sign":
        source = source.replace("from 7", "from 6", 1)
    elif mutation == "wrong_product_sign":
        source = source.replace("from 15", "from 14", 1)
    elif mutation == "shifted_addend":
        source = source.replace("%z from 0", "%z from 1")
    elif mutation == "subtract":
        source = source.replace("comb.add", "comb.sub")
    elif mutation == "nary_add":
        source = source.replace("comb.add %9, %10", "comb.add %9, %10, %10")
    elif mutation == "zero_extension":
        source = source.replace("%1 = comb.replicate %0 : (i1) -> i8", "%1 = hw.constant 0 : i8")
    else:
        source = source.replace("hw.module @Unit(in %x", "hw.module @Unit(in %clock : !seq.clock, in %x")
        source = source.replace(
            "%0 = comb.extract %x", "%saved_x = seq.firreg %x clock %clock : i8\n    %0 = comb.extract %saved_x"
        )
        source = source.replace("comb.concat %1, %x", "comb.concat %1, %saved_x")
    assert source != core(selected)
    facts = generic(source, tools, tmp_path / "defect")
    assert facts["relations"] == [] and facts["unrecognized_outputs"] == 1


def test_source_names_and_operand_order_do_not_assign_meaning(selected, tools, tmp_path):
    source = core(selected).replace("@Unit", "@ArbitrarySourceSymbol").replace("%x", "%left_unknown")
    source = source.replace("comb.add %9, %10", "comb.add %10, %9")
    relation = generic(source, tools, tmp_path / "renamed")["relations"][0]
    assert relation["module"] == "ArbitrarySourceSymbol" and relation["result_bits"] == 20
    assert relation["operands"][0]["input_ordinal"] == 0


@pytest.mark.parametrize("choice", ["unsigned", "floating", "modular"])
def test_incompatible_selected_numeric_semantics_cannot_receive_conditional_constraints(
    selected, tools, tmp_path, choice
):
    facts = generic(core(selected), tools, tmp_path / "semantic-control")
    numeric = {
        "model": {"engine": "integer_reference"},
        "operand_dtype": "int8",
        "accumulator_dtype": "i32",
        "overflow": "bounded_exact",
    }
    if choice == "unsigned":
        numeric["operand_dtype"] = "uint8"
    elif choice == "floating":
        numeric["model"]["engine"] = "specir_fp_reduce"
    else:
        numeric["overflow"] = "modular_wrap"
    rows = O.required_unknowns(facts, spec={"numerical_semantics": numeric}, contraction_owners={"independent-owner"})
    assert rows[0]["requirements"]["compatible_reviewed_owners"] == []
    assert "no compatible" in rows[0]["reason"] and rows[1]["kind"] == "numeric_domain"


def test_native_emitted_bits_match_independent_full_operand_controls(selected, tools, tmp_path):
    destination = tmp_path / "native-bits"
    destination.mkdir()
    original = destination / "Unit.hw.mlir"
    original.write_text(core(selected))
    exported = I.run(
        [str(tools["MERLIN_TEST_CIRCT_OPT"]), "--export-verilog", str(original), "-o", "/dev/null"],
        directory=destination,
        stage="native_arithmetic_control_export",
        inputs=(original,),
        env=_ENV,
        capture_output=True,
        check=True,
        timeout=60,
    )
    source = destination / "Unit.sv"
    source.write_bytes(exported.stdout)
    driver = destination / "control.cpp"
    driver.write_text("""#include "VUnit.h"
#include <cstdint>
#include <cstdio>
int main() {
 VUnit unit; uint64_t count=0, counterexamples=0;
 const int64_t addends[]={-2147483648LL,-524289,-524288,-1,0,1,524287,524288,2147483647};
 for(int64_t a=-128;a<128;++a) for(int64_t b=-128;b<128;++b) for(int64_t c:addends) {
  unit.x=uint32_t(a)&255; unit.y=uint32_t(b)&255; unit.z=uint32_t(c); unit.eval();
  int64_t exact=a*b+c; uint32_t expected=uint64_t(exact)&((1U<<20)-1);
  if(unit.q!=expected) return 2;
  int64_t signed_result=(unit.q&(1U<<19))?int64_t(unit.q)-(1LL<<20):unit.q;
  if(signed_result!=exact) ++counterexamples;
  ++count;
 }
 std::printf("%llu %llu\\n",(unsigned long long)count,(unsigned long long)counterexamples);
 return 0;
}
""")
    model = destination / "model"
    I.run(
        [
            str(tools["MERLIN_TEST_VERILATOR"]),
            "--cc",
            "--exe",
            "--build",
            "-j",
            "2",
            "--top-module",
            "Unit",
            "--Mdir",
            str(model),
            str(source),
            str(driver),
        ],
        directory=destination,
        stage="native_arithmetic_control_build",
        inputs=(source, driver),
        outputs=(model / "VUnit",),
        capture_output=True,
        check=True,
        env=_ENV,
        timeout=120,
    )
    result = I.run(
        [str(model / "VUnit")],
        directory=destination,
        stage="native_arithmetic_control_execute",
        capture_output=True,
        check=True,
        env=_ENV,
        timeout=30,
    )
    count, mismatches_to_unbounded = map(int, result.stdout.split())
    assert count == 256 * 256 * 9 and mismatches_to_unbounded > 0
    for receipt in destination.glob("invocations/*/invocation.json"):
        I.require_environment(receipt, environment=_ENV)


def test_normal_budgeted_generation_retains_real_numeric_path_gaps(automatic, selected, tools, tmp_path):
    # This independently produced scalar unit exposes no transfer interface;
    # request the functional numeric roster without an unrelated transfer goal.
    recipe = yaml.safe_load(automatic["recipe"].read_bytes())
    recipe["component_performance"]["objectives"] = []
    write(automatic["recipe"], recipe)
    intake = R.issue_independent_arithmetic_intake(
        hardware=automatic["hardware_intake"],
        circt_opt=tools["MERLIN_TEST_CIRCT_OPT"],
        forbidden_roots=selected["forbidden_roots"],
        output=tmp_path / "actual-arithmetic",
    )
    policy = yaml.safe_load(automatic["component_coverage"].read_bytes())
    policy.update(schema=A.ARITHMETIC_POLICY_SCHEMA, arithmetic_intake_sha256=intake.sha256)
    write(automatic["component_coverage"], policy)
    automatic["arithmetic_intake"] = intake
    report = automatic_fixtures.run(automatic)
    A.verify(report["automatic_derivation"], report=report)
    missing = report["automatic_derivation"]["required_unknowns"]
    numeric = next(row for row in missing if row["kind"] == "numeric_datapath")
    assert numeric["requirements"]["modular_result_bits"] == 20
    assert numeric["requirements"]["signed_addend_input_bits"] == 32
    assert numeric["requirements"]["compatible_reviewed_owners"] == ["contraction"]
    actual = next(row for row in report["obligations"] if row["id"] == numeric["id"])
    assert actual["mandatory"] is True and actual["state"] == "unavailable" and actual["members"] == []
    assert any(row["selector"] == "rtl_boundary_axis_mapping" for row in missing)
    for obligation in report["obligations"]:
        for member in obligation["members"]:
            path = automatic["output_root"] / member["member"]
            capsule = yaml.safe_load((path / "capsule.yaml").read_bytes())
            inputs = materialize_capsule_leaves(capsule)
            outputs = golden_store.load_golden(path)["outputs"]
            program = capsule["component_program"]
            values = {name: (tensor.shape, list(tensor.data)) for name, tensor in inputs.items()}
            for node in program["nodes"]:
                if node["op"] == "matmul":
                    (lhs_shape, lhs), (rhs_shape, rhs) = [values[name] for name in node["actual_inputs"]]
                    m, k, n = lhs_shape[0], lhs_shape[1], rhs_shape[1]
                    expected = [
                        sum(lhs[i * k + p] * rhs[p * n + j] for p in range(k)) for i in range(m) for j in range(n)
                    ]
                    values[node["name"]] = ((m, n), expected)
                elif node["op"] == "copy":
                    shape, data = values[node["actual_inputs"][0]]
                    values[node["name"]] = (shape, list(data))
                else:
                    raise AssertionError("unexpected source operation in the complete independent control")
            independently_recomputed = {}
            for output in program["outputs"]:
                (m, n), data = values[output["actual_value"]]
                independently_recomputed[output["name"]] = [data[i * n : (i + 1) * n] for i in range(m)]
            assert outputs == independently_recomputed
    changed = copy.deepcopy(report)
    changed["automatic_derivation"]["required_unknowns"].remove(numeric)
    changed["automatic_derivation"]["sha256"] = A.digest(
        {key: value for key, value in changed["automatic_derivation"].items() if key != "sha256"}
    )
    changed["generation_identity"]["automatic_derivation_sha256"] = A.digest(changed["automatic_derivation"])
    with pytest.raises(ValueError, match="source factory"):
        A.verify(changed["automatic_derivation"], report=changed)


def test_v3_requires_original_live_hardware_relation_authority(automatic):
    policy = yaml.safe_load(automatic["component_coverage"].read_bytes())
    policy.update(schema=A.ARITHMETIC_POLICY_SCHEMA, arithmetic_intake_sha256="f" * 64)
    write(automatic["component_coverage"], policy)
    with pytest.raises(ValueError, match="identical live"):
        automatic_fixtures.generation.generate_target("fixture", **automatic)
