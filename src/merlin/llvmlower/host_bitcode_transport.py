"""Explicit LLVM assembly transport for large constant-bearing host modules.

Textual LLVM can exceed Clang's source-location capacity even when the module's
executable code is small. The selected installation's llvm-as serializes the
whole unchanged module to bitcode, without optimization or constant removal.
The ordinary compiler then consumes that observed product with its original
target/ABI/optimization flags. This is transport custody, not numerical proof.
"""

from pathlib import Path

from merlin.common.digest import sha256_file


def assemble_host_bitcode(source, work, *, compiler, compilation, runner):
    """Use an explicitly selected compiler's paired assembler; no fallback."""
    source = Path(source).resolve(strict=True)
    compiler = Path(compiler).resolve(strict=True)
    assembler = compiler.with_name("llvm-as").resolve(strict=True)
    if not source.is_file() or not compiler.is_file() or not assembler.is_file():
        raise ValueError("host bitcode requires ordinary selected source and tool files")
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    bitcode = work / "model.bc"
    compilation.run(
        [assembler, source, "-o", bitcode], runner=runner, inputs=[source], output=bitcode,
    )
    command = compilation.record["commands"][-1]
    if sha256_file(source) != command["inputs"][0]["sha256"]:
        raise ValueError("host LLVM source changed during bitcode assembly")
    return bitcode, {
        "schema": "merlin.host_bitcode_transport.v1",
        "source": command["inputs"][0],
        "bitcode": command["output"],
        "assembler": command["executable"],
        "scope": "whole-module LLVM serialization; no optimization or execution qualification",
    }
