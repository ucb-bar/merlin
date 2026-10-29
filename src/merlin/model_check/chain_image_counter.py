"""Compile counter-only support from an explicitly selected original host CSR.

This reuses a completed program's compiler/ABI flags and source declaration.
It adds no profiling wrappers and establishes no counter frequency or physical
execution origin. The original source and compilation remain independently bound.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from merlin.common import invocation_record

from . import chain_image_layout as layout
from . import chain_image_objects as objects

KIND = "host_source_csr_v1"
SCHEMA = "merlin.chain_counter.host_csr.v1"
SYMBOL = "merlin_chain_host_counter"


def prepare(counter, program, directory, *, nm, timeout):
    if set(counter) != {"kind", "program", "contract"} or counter["kind"] != KIND:
        raise ValueError("source counter selection needs kind, original program and contract")
    contract = json.loads(objects.payload(counter["contract"]))
    if set(contract) != {"schema", "csr", "source"} or contract["schema"] != SCHEMA or contract["csr"] != "mcycle":
        raise ValueError("unsupported original host counter contract")
    command = layout.command(program, "model_main.o")
    sources = [row for row in command["inputs"] if row["path"].endswith(".c")]
    if len(sources) != 1 or any(contract["source"][k] != sources[0][k] for k in ("path", "sha256")):
        raise ValueError("counter source is not the selected program's original compiled host driver")
    objects.reopen(contract["source"])
    original = objects.payload(sources[0]).decode("utf-8")
    if not re.search(r'__asm__\s+volatile\s*\(\s*"csrr %0, mcycle"\s*:\s*"=r"\s*\(\s*\w+\s*\)\s*\)', original):
        raise ValueError("original selected host driver does not declare the supported mcycle read")
    if not any(arg.startswith("-march=rv64") for arg in command["argv"]):
        raise ValueError("source mcycle counter requires the original selected RV64 ABI")
    objects.reopen(command["executable"])
    directory.mkdir()
    source, product = directory / "counter.c", directory / "counter.o"
    source.write_text(
        "/* Counter-only support bound to the original selected host CSR. */\n"
        "#include <stdint.h>\n"
        f"uint64_t {SYMBOL}(void) {{\n"
        '  uint64_t ticks; __asm__ volatile("csrr %0, mcycle" : "=r"(ticks) : : "memory");\n'
        "  return ticks;\n}\n"
    )
    invocation_record.run(
        layout.compile_argv(command, source, product), directory=directory, stage="chain_counter_compile",
        inputs=(*(objects.reopen(row) for row in command["inputs"]), objects.reopen(counter["contract"]), source),
        outputs=(product,), dependencies=(Path(__file__), Path(layout.__file__)),
        capture_output=True, text=True, check=True, timeout=timeout,
    )
    if objects.symbols(nm, product, timeout=timeout, directory=directory).get(SYMBOL) != "T":
        raise ValueError("compiled counter support lacks its actual defining symbol")
    return {
        "kind": KIND, "symbol": SYMBOL, "contract": counter["contract"], "original_source": sources[0],
        "compiler": command["executable"], "source": objects.pin(source), "object": objects.pin(product),
        "invocations": [objects.pin(p) for p in sorted(directory.glob("invocations/*/invocation.json"))],
        "scope": "Original selected mcycle semantics only; no physical counter frequency/execution qualification.",
    }


def verify(record):
    if record is not None:
        for key in ("contract", "original_source", "compiler", "source", "object"):
            objects.reopen(record[key])
        for row in record["invocations"]:
            invocation_record.verify(objects.reopen(row))
