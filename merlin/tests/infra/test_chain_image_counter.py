"""Counter-only support uses actual original source/ABI and observed compilation."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from merlin.llvmlower import toolchain
from merlin.model_check.chain_image_counter import KIND, SCHEMA, SYMBOL, prepare, verify
from merlin.model_check.chain_image_objects import pin
from merlin.model_check.chain_image_sections import symbol_bytes


@pytest.fixture
def original(tmp_path):
    compiler = Path(toolchain.clang()).resolve()
    source, obj = tmp_path / "original.c", tmp_path / "model_main.o"
    source.write_text(
        '#include <stdint.h>\nuint64_t start(void) { uint64_t c0;\n'
        '__asm__ volatile("csrr %0, mcycle" : "=r"(c0)); return c0; }\n'
    )
    argv = [str(compiler), "--target=riscv64-unknown-elf", "-march=rv64gc", "-mabi=lp64d",
            "-O2", "-ffreestanding", "-c", str(source), "-o", str(obj)]
    subprocess.run(argv, capture_output=True, check=True, timeout=15)
    command = {"requested_output": str(obj), "argv": argv, "inputs": [pin(source)],
               "executable": pin(compiler)}
    program = {"recipe": {"commands": [command]}}
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"schema": SCHEMA, "csr": "mcycle", "source": pin(source)}))
    return {"kind": KIND, "program": 0, "contract": pin(contract)}, program, contract


def test_actual_rv64_counter_is_defined_without_instrumenting_original_object(original, tmp_path):
    counter, program, _ = original
    obj = Path(program["recipe"]["commands"][0]["requested_output"])
    before = obj.read_bytes()
    record = prepare(counter, program, tmp_path / "counter", nm=Path(shutil.which("nm")).resolve(), timeout=15)
    verify(record)
    assert obj.read_bytes() == before
    # Actual CSRR a0,mcycle encoding, independently read through ELF section/symbol tables.
    code = symbol_bytes(Path(record["object"]["path"]), SYMBOL)
    assert bytes.fromhex("732500b0") in code
    assert record["invocations"] and "profil" not in record["source"]["path"]
    Path(record["source"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        verify(record)


@pytest.mark.parametrize("change", ["csr", "different_source", "source_changed", "abi", "extra"])
def test_counter_selection_refuses_unbound_source_or_semantics(original, tmp_path, change):
    counter, program, contract = original
    document = json.loads(contract.read_text())
    if change == "csr":
        document["csr"] = "time"
    elif change == "different_source":
        source = tmp_path / "uncompiled.c"
        source.write_bytes(Path(document["source"]["path"]).read_bytes())
        document["source"] = pin(source)
    elif change == "source_changed":
        Path(document["source"]["path"]).write_text("different CSR")
    elif change == "abi":
        program["recipe"]["commands"][0]["argv"] = [
            x for x in program["recipe"]["commands"][0]["argv"] if not x.startswith("-march=")
        ]
    else:
        document["assume_frequency"] = 30000000
    contract.write_text(json.dumps(document))
    counter["contract"] = pin(contract)
    with pytest.raises(ValueError):
        prepare(counter, program, tmp_path / "refused", nm=Path(shutil.which("nm")).resolve(), timeout=15)
    assert not (tmp_path / "refused").exists()
