"""Public Spike service over the existing bare-metal and Zephyr launchers."""

from __future__ import annotations

from .service import RunRequest, RunResult

name = "spike"


def validate(request: RunRequest) -> None:
    if request.arguments:
        raise ValueError("Spike adapter does not support Linux process arguments")
    if request.artifact.kind != "elf":
        raise ValueError("Spike requires a linked ELF artifact")
    with request.artifact.path.open("rb") as stream:
        header = stream.read(64)
    if len(header) < 52 or header[:4] != b"\x7fELF" or header[4] not in (1, 2) or header[5] != 1:
        raise ValueError("Spike input is not a valid ELF header")
    bits = 32 if header[4] == 1 else 64
    if bits == 64 and len(header) < 64:
        raise ValueError("truncated ELF64 header")
    if int.from_bytes(header[18:20], "little") != 243:
        raise ValueError("Spike requires a RISC-V ELF")
    deployment = request.deployment
    if not deployment.isa.startswith(f"rv{bits}"):
        raise ValueError("selected ISA width does not match the ELF")
    abis = {32: {"ilp32": 0, "ilp32f": 2, "ilp32d": 4}, 64: {"lp64": 0, "lp64f": 2, "lp64d": 4, "lp64q": 6}}
    flags_offset = 36 if bits == 32 else 48
    flags = int.from_bytes(header[flags_offset : flags_offset + 4], "little")
    if deployment.abi not in abis[bits] or (flags & 6) != abis[bits][deployment.abi] or flags & 8:
        raise ValueError("selected ABI does not match the supported ELF calling convention")
    if deployment.platform not in {"baremetal", "zephyr"}:
        raise ValueError(f"Spike adapter does not support platform {deployment.platform!r}")
    if deployment.platform == "zephyr" and deployment.dram_base is None:
        raise ValueError("Zephyr Spike execution requires declared DRAM origin and size")


def run(request: RunRequest) -> RunResult:
    validate(request)
    contract = request.deployment
    from .backends.spike import run_elf

    console = run_elf(
        request.artifact.path,
        harts=request.harts,
        isa=contract.isa,
        timeout=request.timeout,
        dram_base=contract.dram_base,
        mem_bytes=contract.memory_bytes,
    )
    return RunResult(
        name,
        request.artifact.sha256,
        "completed",
        console,
        {
            "verification": "not_requested",
            "deployment": contract.system,
            "platform": contract.platform,
            "harts": request.harts,
        },
    )
