"""Static evidence reporting uses the same classes as the actual dispatch audit."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from merlin.targetgen import owned_host_model_execution as owned


@pytest.mark.parametrize("required", [["WORK"], ["MISSING"]])
def test_static_tiers_report_dispatch_classes_without_granting_missing_work(tmp_path, monkeypatch, required):
    from merlin.common import facts_view
    from merlin.compile import model_execution_inputs
    from merlin.llvmlower import compilation_recipe, toolchain
    from merlin.runtime import boards
    from merlin.runtime.backends import spike
    from merlin.targetgen import elf_lanes, native_model_execution
    from merlin.targetgen.rtl import facts as rtl_facts

    elf, ir, obj, facts_path = [tmp_path / name for name in ("fixture.elf", "host.ll", "host.o", "facts.json")]
    for path in (elf, ir, obj):
        path.write_bytes(b"synthetic custody fixture; no target execution")
    facts_path.write_text(json.dumps({"facts": {}}))
    digest = native_model_execution._digest
    compiler = tmp_path / "compiler"
    commands = [
        {"output": {"path": str(obj)}, "argv": ["-c"], "executable": {"path": str(compiler)},
         "inputs": [digest(ir)]},
        {"output": {"path": str(elf)}, "executable": {"path": str(compiler)}, "inputs": [digest(obj)]},
    ]
    monkeypatch.setattr(boards, "load_boards", lambda _: {"board": SimpleNamespace(target="fixture", rtl_sim_config="cfg")})
    monkeypatch.setattr(model_execution_inputs, "selected_firrtl", lambda *a, **k: {})
    monkeypatch.setattr(compilation_recipe, "verify_completed_recipe", lambda *a, **k: {"commands": commands})
    monkeypatch.setattr(toolchain, "clang", lambda: compiler)
    monkeypatch.setattr(spike, "gcc_path", lambda: compiler)
    monkeypatch.setattr(owned, "_pinned", lambda record, role: {"llvm_ir": ir, "object": obj}[role])
    monkeypatch.setattr(facts_view, "interface", lambda *a: {"custom_opcode": 11, "names": {"1": "WORK"}})
    monkeypatch.setattr(elf_lanes, "executable_sections", lambda _: [("text", 0, 4, 0)])
    monkeypatch.setattr(elf_lanes, "instruction_words", lambda *a: [(0, (1 << 25) | 11)])
    monkeypatch.setattr(rtl_facts, "observed_facts", lambda *a, **k: nullcontext())
    record = {"build": {"inputs": {"board_catalog": "fixture", "board": "board"}, "output": {
        "elf": str(elf), "compilation_recipe": {"path": "fixture", "sha256": "fixture"},
        "host_artifacts": {}, "device_artifacts": {"kernels": {}},
    }}}
    dispatch = {"status": "verified", "linked_elf": digest(elf), "source_sha256": "source-pin",
                "observed_classes": ["WORK", "CONFIG", "WORK"]}
    policy = {"expected_instruction_classes": required, "prohibited_instruction_roles": []}
    def audit():
        return owned._static_tiers({}, record, target="fixture", facts={"path": str(facts_path)},
                                   dispatch=dispatch, policy=policy)
    if required == ["MISSING"]:
        with pytest.raises(ValueError, match="omit frozen required instruction classes"):
            audit()
    else:
        result = audit()
        assert result["L1"]["observed_classes"] == ["CONFIG", "WORK"]
        assert result["L1"]["required_classes"] == required
        assert result["L1"]["whole_elf_known_custom_commands"] == 1
        assert result["L0"]["source_sha256"] == "source-pin"
        assert set(result) == {"L0", "L1"}  # Static checks cannot grant numerical tiers.
