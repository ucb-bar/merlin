"""RoCC transport delegates operand semantics to the selected support provider."""

from types import SimpleNamespace

import pytest

from merlin.runtime.backends import base
from merlin.targetgen.rocc import asm, decode


class SyntheticSemantics:
    """Two intentionally incompatible layouts without any accelerator dependencies."""

    def __init__(self, name, shift, funct):
        self.name = name
        self.shift = shift
        self.funct = funct
        self.opcode = 0x2B
        self.calls = []

    def isa_constants(self, target):
        self.calls.append(target)
        return {"CUSTOM_OPCODE": self.opcode, "FUNCT3": 3, "FUNCT_CLASS": {self.funct: self.name}, "SHIFT": self.shift}

    def decode_instruction(self, funct, rs1, rs2, isa):
        if funct not in isa["FUNCT_CLASS"]:
            return "UNKNOWN", {}
        raw = rs1["raw"]
        return isa["FUNCT_CLASS"][funct], {
            "selector": None if raw is None else (raw >> isa["SHIFT"]) & 7,
            "rhs": dict(rs2),
        }

    def instruction_funct(self, name, rs1, isa):
        if name != self.name:
            raise ValueError(f"unsupported synthetic instruction {name}")
        if (rs1 >> isa["SHIFT"]) & 7 != 5:
            raise ValueError("synthetic selector must equal five")
        return self.funct


@pytest.fixture
def providers(monkeypatch):
    selected = {
        "first": SimpleNamespace(rocc_semantics=SyntheticSemantics("SEND", 4, 9)),
        "second": SimpleNamespace(rocc_semantics=SyntheticSemantics("ISSUE", 12, 17)),
    }
    monkeypatch.setattr(base, "get_backend", lambda target: selected[target])
    return selected


def program(funct=9, rs1=80, rs2=37):
    return f"""module {{
  llvm.func @kernel() {{
    %a = llvm.mlir.constant({rs1} : i64) : i64
    %b = llvm.mlir.constant({rs2} : i64) : i64
    llvm.inline_asm has_side_effects ".insn r 0x2b, 3, {funct}, x0, $0, $1", "r,r" %a, %b : (i64, i64) -> ()
    llvm.return
  }}
}}"""


@pytest.mark.parametrize(
    "target,funct,raw,name",
    [
        ("first", 9, 5 << 4, "SEND"),
        ("second", 17, 5 << 12, "ISSUE"),
    ],
)
@pytest.mark.parametrize("fallback", [False, True])
def test_provider_layout_applies_to_both_decode_paths(providers, target, funct, raw, name, fallback):
    text = program(funct, raw)
    if fallback:
        text = "not a module\n" + text
    trace = decode.decode_text(text, target=target)
    (instruction,) = trace["instructions"]
    assert instruction["class"] == name
    assert instruction["decoded"]["selector"] == 5
    assert instruction["decoded"]["rhs"]["raw"] == 37
    assert providers[target].rocc_semantics.calls


def test_missing_capability_refuses_even_after_success(providers):
    text = program()
    decode.decode_text(text, target="first")
    providers["first"] = SimpleNamespace()
    with pytest.raises((NotImplementedError, ValueError, TypeError, AttributeError), match="rocc|RoCC|semantics"):
        decode.decode_text(text, target="first")


def test_changed_facts_are_not_hidden_by_cache(providers):
    text = program()
    before = decode.decode_text(text, target="first")
    providers["first"].rocc_semantics.shift = 5
    after = decode.decode_text(text, target="first")
    assert before["instructions"][0]["decoded"]["selector"] == 5
    assert after["instructions"][0]["decoded"]["selector"] == 2


def test_changed_callback_is_not_hidden_by_equal_constants(providers):
    text = program()
    decode.decode_text(text, target="first")
    providers["first"].rocc_semantics.decode_instruction = lambda funct, rs1, rs2, isa: ("REPLACED", {"new": True})
    result = decode.decode_text(text, target="first")
    assert result["instructions"][0]["class"] == "REPLACED"


def test_provider_selection_checked_even_after_success(providers, monkeypatch):
    text = program()
    decode.decode_text(text, target="first")

    def refuse(target):
        raise base.PluginOwnershipError("selected provider changed")

    monkeypatch.setattr(base, "get_backend", refuse)
    with pytest.raises(base.PluginOwnershipError, match="selected provider changed"):
        decode.decode_text(text, target="first")


def test_nested_result_mutation_does_not_poison_later_decode(providers):
    text = program()
    first = decode.decode_text(text, source="first", target="first")
    first["instructions"][0]["decoded"]["rhs"]["raw"] = -100
    first["instructions"].append({"class": "CORRUPTED"})
    second = decode.decode_text(text, source="second", target="first")
    assert second["source"] == "second"
    (instruction,) = second["instructions"]
    assert instruction["decoded"]["rhs"]["raw"] == 37


@pytest.mark.parametrize("target,name,raw", [("first", "SEND", 5 << 4), ("second", "ISSUE", 5 << 12)])
def test_assembly_uses_provider_mapping_and_validation(providers, target, name, raw):
    text = asm.assemble_program(target, [(name, raw, 37)], kernel_symbol="kernel")
    (instruction,) = decode.decode_text(text, target=target)["instructions"]
    assert instruction["class"] == name
    assert instruction["decoded"]["selector"] == 5
    with pytest.raises(asm.AsmError, match="selector"):
        asm.assemble_program(target, [(name, 0, 37)], kernel_symbol="kernel")
    with pytest.raises(asm.AsmError, match="unsupported"):
        asm.assemble_program(target, [("UNSUPPORTED", raw, 37)], kernel_symbol="kernel")


def test_real_registry_loads_distinct_external_semantics(tmp_path):
    import inspect
    import os
    import subprocess
    import sys

    for target, name, shift, funct in (("first", "SEND", 4, 9), ("second", "ISSUE", 12, 17)):
        root = tmp_path / target
        (root / "contracts").mkdir(parents=True)
        (root / "contracts/target_contract.yaml").write_text(f"name: {target}\nplugin:\n  backend: backend\n")
        (root / "backend.py").write_text(
            "from merlin.runtime.backends.base import BackendInfo, BackendKind, TargetClass, register\n"
            + inspect.getsource(SyntheticSemantics)
            + f"\nrocc_semantics = SyntheticSemantics({name!r}, {shift}, {funct})\n"
            + f"register(BackendInfo({target!r}, TargetClass.NPU, BackendKind.KERNEL, __name__))\n"
        )
    env = dict(os.environ)
    env.pop("MERLIN_TARGET_CONTRACT", None)
    env["MERLIN_TARGET_PATH"] = str(tmp_path)
    code = """
from merlin.targetgen.rocc import asm, decode
for target, name, shift in (("first", "SEND", 4), ("second", "ISSUE", 12)):
    text = asm.assemble_program(target, [(name, 5 << shift, 37)], kernel_symbol="kernel")
    instruction, = decode.decode_text(text, target=target)["instructions"]
    assert instruction["class"] == name
    assert instruction["decoded"]["selector"] == 5
print("two external providers loaded")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert "two external providers loaded" in result.stdout


@pytest.mark.parametrize("width", [16, 32, 64])
def test_manifest_does_not_infer_accelerator_flags_from_address_width(tmp_path, monkeypatch, width):
    from merlin.targetgen.target_experiment import load_capability_manifest

    def refuse(*args):
        pytest.fail("reading a manifest must not load executable backend code")

    monkeypatch.setattr(base, "get_backend", refuse)
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        "name: unrelated\ncompute_units: [{name: lanes, kind: vector, dtypes: [int8]}]\n"
        f"encoding: {{addr_len: {width}, custom_field: 7}}\n"
    )
    manifest = load_capability_manifest("unrelated", contract_path=contract)
    assert manifest.encoding == {"addr_len": width, "custom_field": 7}
    contract.write_text(contract.read_text().replace("custom_field: 7", "readout_bits: {custom_flag: 5}"))
    assert load_capability_manifest("unrelated", contract_path=contract).encoding == {
        "addr_len": width,
        "readout_bits": {"custom_flag": 5},
    }


def test_isa_emission_uses_provider_encoding_and_propagates_errors(providers, monkeypatch, capsys):
    from merlin.targetgen import target_experiment
    from merlin.targetgen.rtl import gen_isa_module

    declared = {"addr_len": 16}
    monkeypatch.setattr(target_experiment, "load_capability_manifest", lambda _: SimpleNamespace(encoding=declared))
    monkeypatch.setattr(gen_isa_module, "load_facts", lambda _: {})
    monkeypatch.setattr(gen_isa_module, "generate", lambda facts, encoding: str(encoding["custom_flag"]))
    providers["first"].rocc_semantics.encoding_fields = lambda encoding: {**encoding, "custom_flag": 123}
    assert gen_isa_module.main(["--target", "first"]) == 0
    assert capsys.readouterr().out.strip() == "123"
    assert declared == {"addr_len": 16}

    def refuse(encoding):
        raise ValueError("cannot derive declared encoding")

    providers["first"].rocc_semantics.encoding_fields = refuse
    with pytest.raises(ValueError, match="cannot derive"):
        gen_isa_module.main(["--target", "first"])
    assert not capsys.readouterr().out
