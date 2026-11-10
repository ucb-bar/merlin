"""Private operator link control for the selected public callable ABI.

The test implementation is never an author input or a submitted backend. It
tests C prototypes and selected native linkage only, with no target instruction,
kernel algorithm, simulator, numerical result or performance claim.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml

from merlin.development.backend import public_runtime_abi as P


def _pin(record):
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("native ABI link control requires an exact input pin")
    path = P._regular(record["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("native ABI link control input differs from selected pin")
    return path


def _prototype(row):
    if row["role"] == "entry" and row.get("parameters") == "per_case_logical_tensor_pointers":
        return f"void {row['symbol']}(void *arg_0)"
    if row["role"] == "completion" and row.get("parameters") == []:
        return f"void {row['symbol']}(void)"
    raise ValueError("native ABI link control has an unsupported callable prototype")


def render(contract, protocol):
    """Caller comes from private authority; definitions come only from public data."""
    required = P.selected_callables(contract)
    public = (protocol.get("logical_kernel_abi") or {}).get("required_callables")
    if not isinstance(public, list):
        raise ValueError("public protocol does not declare its required callable list")
    # Validate public names independently of selected names before emitting C.
    for row in public:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("symbol"), str)
            or not row["symbol"].isascii()
            or not row["symbol"].isidentifier()
            or row.get("return_type") != "void"
            or row.get("linkage") != "C"
            or row.get("defined_by") != "candidate_package"
        ):
            raise ValueError("public ABI link control cannot render an invalid callable")
    if len({row["symbol"] for row in public}) != len(public):
        raise ValueError("public ABI link control refuses duplicate callable definitions")
    # Public prototypes and original caller prototypes share a C translation
    # unit: conflicting signatures fail at compile time, missing public
    # definitions fail at link time. Ordinary cross-module C linkage alone
    # cannot establish prototype compatibility.
    declarations = [f"extern {_prototype(row)};" for row in public]
    declarations += [f"extern {_prototype(row)};" for row in required]
    private_symbols = {row["symbol"] for row in required}
    start = "merlin_public_abi_link_control"
    while start in private_symbols | {row["symbol"] for row in public}:
        start += "_"
    calls = [
        f"  {row['symbol']}((void *)0);" if row["role"] == "entry" else f"  {row['symbol']}();" for row in required
    ]
    caller = "\n".join([*declarations, f"void {start}(void) {{", *calls, "}"]) + "\n"
    definitions = "\n".join(_prototype(row) + " { }" for row in public) + "\n"
    return caller, definitions, start


def probe(contract, protocol, *, output, timeout=30):
    """Cross-link a private control using the original selected toolchain pins."""
    caller, definitions, start = render(contract, protocol)
    runner = contract.get("runner") or {}
    toolchain = (runner.get(runner.get("backend"), {}) or {}).get("toolchain") or {}
    compiler = _pin(toolchain.get("compiler"))
    cflags, ldflags = toolchain.get("cflags", []), toolchain.get("ldflags", [])
    if (
        not isinstance(cflags, list)
        or not isinstance(ldflags, list)
        or any(not isinstance(flag, str) for flag in [*cflags, *ldflags])
    ):
        raise ValueError("native ABI link control requires declared compiler/linker flags")
    script = _pin(toolchain["link_script"]) if toolchain.get("link_script") is not None else None
    output = Path(output).absolute()
    if output.resolve() != output or output.exists():
        raise ValueError("native ABI link control needs fresh canonical private output")
    output.mkdir(parents=True, mode=0o700)
    source, implementation, elf = output / "caller.c", output / "public-definitions.c", output / "control.elf"
    source.write_text(caller)
    implementation.write_text(definitions)
    command = [
        str(compiler),
        *cflags,
        "-fno-lto",
        str(source),
        str(implementation),
        "-nostdlib",
        "-nostartfiles",
        *ldflags,
        *(["-T", str(script)] if script is not None else []),
        "-Wl,-e," + start,
        "-o",
        str(elf),
    ]
    process = subprocess.run(command, capture_output=True, timeout=timeout)
    (output / "stdout.log").write_bytes(process.stdout)
    (output / "stderr.log").write_bytes(process.stderr)
    # Recheck the tool identity after execution; successful linkage never
    # certifies a different compiler or linker script selected concurrently.
    _pin(toolchain["compiler"])
    if script is not None:
        _pin(toolchain["link_script"])
    result = dict(
        schema="merlin.private_public_runtime_abi_link_control.v1",
        status="pass" if process.returncode == 0 and elf.is_file() else "fail",
        returncode=process.returncode,
        command=command,
        selected_callables=P.selected_callables(contract),
        public_callables=protocol["logical_kernel_abi"]["required_callables"],
        caller_sha256=hashlib.sha256(caller.encode()).hexdigest(),
        public_definitions_sha256=hashlib.sha256(definitions.encode()).hexdigest(),
        elf_sha256=hashlib.sha256(elf.read_bytes()).hexdigest() if elf.is_file() else None,
        authority="prototype/linkage control only; no execution or kernel implementation credit",
        author_artifacts_delivered=False,
    )
    (output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def verify_and_probe(bundle, *, workspace, repo, output, timeout=30):
    """Use actual frozen native input ownership; never live source fallbacks."""
    verified = P.verify(bundle, workspace=workspace, repo=repo)
    output = Path(output).absolute()
    if output.resolve().is_relative_to(Path(workspace).resolve()):
        raise ValueError("native ABI link control must stay outside the author workspace")
    from merlin.targetgen.sandbox.bwrap import snapshot_input_paths

    paths = snapshot_input_paths(
        Path(workspace), bundle, [Path(verified[k]["path"]) for k in ("target_contract", "protocol")], repo=Path(repo)
    )
    contract, protocol = [yaml.safe_load(P._regular(path).read_bytes()) for path in paths]
    return probe(contract, protocol, output=output, timeout=timeout)
