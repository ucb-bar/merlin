"""Instruction interfaces are generated from independent synthetic declarations."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json

import pytest
import yaml
from xdsl.context import Context
from xdsl.dialects.builtin import Builtin, IntAttr, StringAttr
from xdsl.dialects.func import Func
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import MemoryEffect
from xdsl.utils.exceptions import VerifyException

from merlin.targetgen.isa_codegen_cli import inspect_inputs, main
from merlin.targetgen.isa_dialect_gen import decode_instruction, encode_instruction, generate_package
from merlin.targetgen.isa_spec import IsaSpecError, resolve_isa, resolved_isa_from_record, validate_isa_spec


def _ref(*parts):
    return {"fact": list(parts)}


def _inputs(variant: int = 0):
    target = ("synthetic_alpha", "synthetic_beta")[variant]
    namespace = ("alpha", "beta")[variant]
    facts = {
        "facts": {
            "target": target,
            "word_bits": (12, 16)[variant],
            "data_bits": (8, 16)[variant],
            "alignment": (1, 2)[variant],
            "fields": [
                {"offset": 0, "width": 2, "value": (1, 2)[variant]},
                {"offset": 2, "width": (3, 5)[variant]},
                {"offset": (5, 7)[variant], "width": (3, 5)[variant]},
                {"offset": (8, 12)[variant], "width": 4},
            ],
            "geometry": {"rows": (3, 7)[variant], "cols": (5, 2)[variant]},
        }
    }
    raw = json.dumps(facts, sort_keys=True).encode()
    digest = hashlib.sha256(raw).hexdigest()

    def arg(name, signed=False):
        return {
            "name": name,
            "type": {"kind": "integer", "bits": _ref("facts", "data_bits"), "signed": signed},
            "unit": "register_index",
            "alignment": _ref("facts", "alignment"),
        }

    fields = [
        {
            "offset": _ref("facts", "fields", index, "offset"),
            "width": _ref("facts", "fields", index, "width"),
            **(
                {"fixed": _ref("facts", "fields", index, "value")}
                if index == 0
                else {"argument": ("lhs", "rhs", "shift")[index - 1]}
            ),
        }
        for index in range(4)
    ]
    spec = {
        "schema": "merlin.isa_spec.v1",
        "namespace": namespace,
        "facts_sha256": digest,
        "word_bits": _ref("facts", "word_bits"),
        "resources": [
            {"name": "configuration", "kind": "state", "unit": "state_value"},
            {"name": "accumulator", "kind": "memory", "unit": "rows", "capacity": _ref("facts", "geometry", "rows")},
        ],
        "instructions": [
            {
                "name": "step",
                "operands": [arg("lhs"), arg("rhs")],
                "results": [],
                "immediates": [arg("shift", True)],
                "encoding": fields,
                "semantics": {
                    "primitive": "integer.add",
                    "bindings": {"left": "lhs", "right": "rhs"},
                    "parameters": {"overflow": "wrap"},
                },
                "effects": {"reads": ["configuration"], "writes": ["accumulator"]},
            }
        ],
    }
    return target, spec, facts, digest, raw


def _model(variant=0):
    target, spec, facts, digest, _ = _inputs(variant)
    return resolve_isa(spec, facts, target=target, facts_sha256=digest)


def test_readiness_reports_missing_declaration_without_generating(tmp_path, capsys):
    target, spec, facts, digest, raw = _inputs()
    sw = tmp_path / 'software.yaml'
    fp = tmp_path / 'facts.json'
    fp.write_bytes(raw)
    sw.write_text(yaml.safe_dump({'schema': 'merlin.software_spec.v1', 'target': target}))
    assert main(['--software-spec', str(sw), '--facts', str(fp), '--target', target, '--inspect']) == 2
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'missing_declaration'
    assert not report['generation_ready']
    assert report['facts_sha256'] == digest
    assert report['compiler_execution'] == 'UNKNOWN'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['facts.json', 'software.yaml']


def test_readiness_resolves_declaration_but_does_not_claim_rtl_or_execution(tmp_path):
    target, spec, facts, digest, raw = _inputs()
    sw = tmp_path / 'software.yaml'
    fp = tmp_path / 'facts.json'
    fp.write_bytes(raw)
    sw.write_text(yaml.safe_dump({'schema': 'merlin.software_spec.v1', 'target': target, 'isa': spec}))
    report = inspect_inputs(software_spec=sw, facts=fp, target=target)
    assert report['generation_ready']
    assert report['model_sha256'] == _model().identity()
    assert report['independent_rtl_correspondence'] == report['compiler_execution'] == 'UNKNOWN'
    spec['facts_sha256'] = '0' * 64
    sw.write_text(yaml.safe_dump({'schema': 'merlin.software_spec.v1', 'target': target, 'isa': spec}))
    invalid = inspect_inputs(software_spec=sw, facts=fp, target=target)
    assert invalid['status'] == 'invalid' and not invalid['generation_ready']


def _load_module(tmp_path, model):
    generate_package(model, tmp_path)
    module_spec = importlib.util.spec_from_file_location(
        f"generated_{model.namespace}", tmp_path / "instruction_dialect.py"
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _program(model, shift=2, dtype=None):
    bits = model.instructions[0].operands[0].type.bits
    immediate_bits = model.instructions[0].immediates[0].type.bits
    dtype = dtype or f"i{bits}"
    return f'''builtin.module {{
      func.func @test(%a: {dtype}, %b: {dtype}) {{
        "{model.namespace}.step"(%a, %b) {{shift = {shift} : i{immediate_bits}}} : ({dtype}, {dtype}) -> ()
        func.return
      }}
    }}'''


@pytest.mark.parametrize("variant", [0, 1])
def test_actual_generated_import_ir_roundtrip_and_codec(tmp_path, variant):
    model = _model(variant)
    module = _load_module(tmp_path, model)
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)
    ctx.load_dialect(module.DIALECT)
    parsed = Parser(ctx, _program(model)).parse_module()
    parsed.verify()
    from io import StringIO

    output = StringIO()
    Printer(stream=output).print_op(parsed)
    again = Parser(ctx, output.getvalue()).parse_module()
    again.verify()
    assert parsed.is_structurally_equivalent(again)
    values = {"lhs": 2, "rhs": 4, "shift": -2}
    word = encode_instruction(model, "step", values)
    assert decode_instruction(model, word) == ("step", values)
    assert module.decode(module.encode("step", values)) == ("step", values)
    op = next(op for op in parsed.walk() if op.name == f"{model.namespace}.step")
    effects = op.get_trait(MemoryEffect).get_effects(op)
    assert {effect.resource.name() for effect in effects} == {
        f"{model.namespace}:configuration",
        f"{model.namespace}:accumulator",
    }
    registry = json.loads((tmp_path / "isa-resolved.json").read_text())
    assert registry["identity"] == model.identity() == module.MODEL_SHA256
    assert json.loads(registry["instructions"][0]["semantics_json"])["primitive"] == "integer.add"
    assert registry["resources"][1]["capacity"] == (3, 7)[variant]
    assert generate_package(model, tmp_path)["model_sha256"] == model.identity()


@pytest.mark.parametrize("shift,dtype", [(8, None), (1, "i32")])
def test_instruction_verifier_rejects_invalid_immediates_and_types(tmp_path, shift, dtype):
    model = _model()
    module = _load_module(tmp_path, model)
    ctx = Context()
    for dialect in (Builtin, Func, module.DIALECT):
        ctx.load_dialect(dialect)
    with pytest.raises(VerifyException):
        Parser(ctx, _program(model, shift=shift, dtype=dtype)).parse_module().verify()


def test_unknown_effects_do_not_become_pure(tmp_path):
    target, spec, facts, digest, _ = _inputs()
    spec["instructions"][0].pop("effects")
    model = resolve_isa(spec, facts, target=target, facts_sha256=digest)
    module = _load_module(tmp_path, model)
    trait = module.INSTRUCTIONS["step"].get_trait(MemoryEffect)
    assert trait.get_effects(module.INSTRUCTIONS["step"].create()) is None


@pytest.mark.parametrize(
    "problem", ["stale", "target", "missing", "overlap", "overflow", "gap", "ambiguous", "literal"]
)
def test_resolution_refuses_bad_evidence_and_encodings(problem):
    target, spec, facts, digest, _ = _inputs()
    if problem == "stale":
        digest = "0" * 64
    elif problem == "target":
        target = "unselected"
    elif problem == "missing":
        del facts["facts"]["word_bits"]
    elif problem == "overlap":
        facts["facts"]["fields"][1]["offset"] = 0
    elif problem == "overflow":
        facts["facts"]["fields"][0]["value"] = 4
    elif problem == "gap":
        facts["facts"]["word_bits"] += 1
    elif problem == "ambiguous":
        other = copy.deepcopy(spec["instructions"][0])
        other["name"] = "other"
        spec["instructions"].append(other)
    elif problem == "literal":
        spec["instructions"][0]["encoding"][0]["fixed"] = 1
    with pytest.raises(IsaSpecError):
        resolve_isa(spec, facts, target=target, facts_sha256=digest)


@pytest.mark.parametrize(
    "values",
    [
        {"lhs": 8, "rhs": 0, "shift": 0},
        {"lhs": 0, "rhs": 0, "shift": -9},
        {"lhs": 0, "rhs": 0},
        {"lhs": 0, "rhs": 0, "shift": False},
    ],
)
def test_encoder_requires_exact_valid_bindings(values):
    with pytest.raises(IsaSpecError):
        encode_instruction(_model(), "step", values)


def test_different_geometry_alignment_and_illegal_decode():
    model = _model(1)
    with pytest.raises(IsaSpecError, match="misaligned"):
        encode_instruction(model, "step", {"lhs": 1, "rhs": 2, "shift": 0})
    with pytest.raises(IsaSpecError):
        decode_instruction(model, 0)
    with pytest.raises(IsaSpecError):
        decode_instruction(model, 1 << model.word_bits)


def test_cli_uses_explicit_selected_facts_and_never_overwrites_model(tmp_path):
    target, spec, _, _, raw = _inputs()
    spec_path = tmp_path / "software.yaml"
    facts_path = tmp_path / "facts.json"
    spec_path.write_text(yaml.safe_dump({"schema": "merlin.software_spec.v1", "target": target, "isa": spec}))
    facts_path.write_bytes(raw)
    out = tmp_path / "generated"
    assert (
        main(["--software-spec", str(spec_path), "--facts", str(facts_path), "--target", target, "--out", str(out)])
        == 0
    )
    other = _model(1)
    with pytest.raises(IsaSpecError, match="different generated"):
        generate_package(other, out)


def test_optional_declaration_validation_is_conservative():
    _, spec, _, _, _ = _inputs()
    validate_isa_spec(spec)
    spec["instructions"][0]["semantics"]["bindings"]["left"] = "undeclared"
    with pytest.raises(IsaSpecError):
        validate_isa_spec(spec)


def test_address_space_types_and_results_are_real_ir_constraints(tmp_path):
    target, spec, facts, digest, _ = _inputs()
    instruction = spec["instructions"][0]
    instruction["operands"][0]["type"] = {
        "kind": "address",
        "space": "accumulator",
        "bits": _ref("facts", "data_bits"),
        "signed": False,
    }
    instruction["results"] = [
        {
            "name": "value",
            "type": {
                "kind": "integer",
                "bits": _ref("facts", "data_bits"),
                "signed": True,
            },
            "unit": "scalar",
        }
    ]
    model = resolve_isa(spec, facts, target=target, facts_sha256=digest)
    module = _load_module(tmp_path, model)
    ctx = Context()
    for dialect in (Builtin, Func, module.DIALECT):
        ctx.load_dialect(dialect)
    address_type = str(module.AddressType(StringAttr("accumulator"), IntAttr(8)))
    program = f"""builtin.module {{
      func.func @read(%a: {address_type}, %b: i8) -> i8 {{
        %v = "alpha.step"(%a, %b) {{shift = 0 : i8}} : ({address_type}, i8) -> i8
        func.return %v : i8
      }}
    }}"""
    Parser(ctx, program).parse_module().verify()
    with pytest.raises(VerifyException):
        Parser(ctx, program.replace('"accumulator"', '"different"')).parse_module().verify()


@pytest.mark.parametrize("mutation", ["overlap", "unknown_effect", "stale_identity", "oversized", "missing"])
def test_rehydration_checks_generated_model_structure(mutation):
    model = _model()
    record = model.record()
    if mutation == "overlap":
        record["instructions"][0]["fields"][1]["offset"] = 0
    elif mutation == "unknown_effect":
        record["instructions"][0]["effects_json"] = json.dumps({"reads": ["undeclared"], "writes": []})
    elif mutation == "stale_identity":
        record["identity"] = "0" * 64
    elif mutation == "oversized":
        record["instructions"][0]["fields"][1]["width"] = 1 << 30
    else:
        del record["namespace"]
    with pytest.raises(IsaSpecError):
        resolved_isa_from_record(record)


def test_software_spec_optional_isa_validation_preserves_existing_inputs():
    from merlin.targetgen.software_spec import validate_software_spec

    target, isa, _, _, _ = _inputs()
    software = {
        "schema": "merlin.software_spec.v1",
        "target": target,
        "numerical_semantics": {
            "model": {"engine": "integer_reference"},
            "operand_dtype": "int8",
            "accumulator_dtype": "i32",
            "readout_dtype": "i32",
            "subnormal_operand_flush": False,
        },
        "operations": {
            "contraction": {
                "placement": "accelerator",
                "operand_dtypes": ["int8"],
                "ranks": [2],
            }
        },
    }
    assert "isa" not in validate_software_spec(software)
    software["isa"] = isa
    assert validate_software_spec(software)["isa"] == isa
    isa["word_bits"] = 8
    with pytest.raises(IsaSpecError):
        validate_software_spec(software)


def test_unsigned_immediate_interprets_signless_attribute_bits(tmp_path):
    target, spec, facts, digest, _ = _inputs()
    immediate = spec["instructions"][0]["immediates"][0]
    immediate["type"] = {"kind": "integer", "bits": 4, "signed": False}
    model = resolve_isa(spec, facts, target=target, facts_sha256=digest)
    module = _load_module(tmp_path, model)
    ctx = Context()
    for dialect in (Builtin, Func, module.DIALECT):
        ctx.load_dialect(dialect)
    Parser(ctx, _program(model, shift=15)).parse_module().verify()
    values = {"lhs": 2, "rhs": 4, "shift": 15}
    assert module.decode(module.encode("step", values)) == ("step", values)
