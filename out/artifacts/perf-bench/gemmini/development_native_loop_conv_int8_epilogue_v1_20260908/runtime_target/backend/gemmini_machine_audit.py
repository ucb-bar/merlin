"""Cheap post-LLVM structural feedback through the existing target object pipeline.

Never executes generated code. Instruction sites and object bytes are not dynamic
work or cycles; this catches a pre-LLVM win that becomes a downstream tradeoff.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from time import monotonic


def _instruction_sites(disassembly: str, audit) -> dict:
    """Count explicit encoded-but-undecoded sites instead of silently dropping them."""
    from merlin.baselines.rvv_audit import _insn_mnemonic
    unknown = 0
    for line in disassembly.splitlines():
        address, colon, rest = line.partition(":")
        if not colon or not address.strip() or any(c not in "0123456789abcdefABCDEF" for c in address.strip()):
            continue
        raw, tab, text = rest.lstrip().partition("\t")
        encoded = "".join(raw.split())
        if (tab and text.strip() and encoded and len(encoded) % 2 == 0
                and all(c in "0123456789abcdefABCDEF" for c in encoded)
                and _insn_mnemonic(line) is None):
            unknown += 1
    result = audit.instruction_mix()
    result["schema"] = "encoded_instruction_sites_v1"
    result["decoded_total"] = result["total"]
    result["undecoded"] = unknown
    result["total"] += unknown
    result["other"] += unknown
    for name in ("vector", "vsetvl", "scalar_int", "scalar_float", "other"):
        result[name + "_frac"] = result[name] / result["total"] if result["total"] else 0.0
    return result


def machine_artifact_policy_identity() -> dict:
    """Resolve the actual build policy before reusing an immutable baseline audit."""
    from merlin.baselines import rvv_audit
    from merlin.llvmlower import codegen, toolchain
    from merlin.runtime.backends import base
    from . import gemmini

    def identified(path):
        # A sandbox may grant the snapshot's symlink spelling without granting
        # its host-side canonical target. Hash through the link, but execute
        # the granted spelling; canonicalization here breaks an otherwise exact policy.
        path = Path(path).absolute()
        return {"path": str(path), "resolved_path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    # Use the same LLVM install/public spelling as the object pipeline, not a
    # different board's convenience disassembler outside this target's grants.
    disassembler = toolchain.objdump()
    if not disassembler.is_file():
        raise ValueError("machine instruction audit requires the target disassembler")
    return {
        "schema": "machine_artifact_policy_identity_v1",
        "flags": [*codegen.RISCV_FLAGS, base.harness_build_recipe("gemmini").march()],
        "compiler": identified(toolchain.clang()),
        "translator": identified(toolchain.mlir_translate()),
        "disassembler": identified(disassembler),
        "implementations": {name: identified(path) for name, path in (
            ("adapter", __file__), ("target_recipe", gemmini.__file__),
            ("recipe_interface", base.__file__), ("codegen", codegen.__file__),
            ("toolchain", toolchain.__file__), ("instruction_audit", rvv_audit.__file__))},
    }


def analyze_machine_artifact(lowered_text: str, *, workdir: Path,
                             run_command: Callable,
                             timeout_seconds: float = 60) -> dict:
    """The host must supply its answer-masked runner, including for the assembler.

    Not executing a kernel is insufficient isolation: assembler directives can
    read files. There is deliberately no direct subprocess fallback here.
    """
    from merlin.baselines.rvv_audit import classify_disasm

    if not 0 < timeout_seconds <= 600:
        raise ValueError("machine artifact compilation must fit the remaining iteration budget")
    started = monotonic()
    deadline = started + timeout_seconds
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=False)
    source, ll, obj = (work / name for name in ("source.mlir", "target.ll", "target.o"))
    source.write_text(lowered_text)
    policy = machine_artifact_policy_identity()
    compiler, translator, disassembler = (policy[name]["path"] for name in
                                          ("compiler", "translator", "disassembler"))
    flags = policy["flags"]

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def remaining():
        value = deadline - monotonic()
        if value <= 0:
            raise TimeoutError("machine artifact audit exhausted its iteration budget")
        return value

    commands = ([str(translator), "--mlir-to-llvmir", str(source), "-o", str(ll)],
                [str(compiler), *flags, "-c", str(ll), "-o", str(obj)])
    for index, argv in enumerate(commands):
        result = run_command(argv, timeout_s=remaining())
        (work / f"build_{index}.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise ValueError("machine artifact compilation failed: " + result.stderr[-2000:])
    result = run_command([str(disassembler), "-d", str(obj)], timeout_s=remaining())
    if result.returncode:
        raise ValueError("machine artifact disassembly failed: " + result.stderr[-2000:])
    audit = classify_disasm(result.stdout, source=str(obj))
    sites = _instruction_sites(result.stdout, audit)
    if sites["total"] <= 0:
        raise ValueError("machine artifact has no encoded instruction sites; missing is not zero work")
    return {
        "schema": "compiled_machine_activity_v1", "status": "compiled",
        "source_sha256": sha(source), "llvm_sha256": sha(ll), "object_sha256": sha(obj),
        "object_bytes": obj.stat().st_size, "instruction_sites": sites,
        "instruction_decode_complete": sites["undecoded"] == 0,
        "compiler": {"path": str(compiler), "sha256": sha(compiler), "flags": flags},
        "translator": {"path": str(translator), "sha256": sha(translator)},
        "disassembler": {"path": str(disassembler), "sha256": sha(disassembler)},
        "elapsed_seconds": monotonic()-started,
        "build_policy_identity": policy,
        "full_model_executed": False, "timing_measured": False,
        "licence": "post-LLVM encoded sites, with undecoded sites explicit; not ISA compatibility, dynamic instructions, traffic or cycles",
    }
