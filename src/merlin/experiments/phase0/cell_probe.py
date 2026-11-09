"""Deterministic raw-bit characterization of an explicitly selected RTL cell.

The caller supplies independent expected values and a finite domain. No target,
arithmetic semantics, or compiler claim is inferred by this process runner.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from merlin.common.arrival_stamp import stream_stamped
from merlin.common.digest import is_sha256
from merlin.targetgen.rtl.extract_module import extract


def _identity(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _simple_identifier(value: str) -> bool:
    """Accept only the ASCII identifier subset used for native C++ ports."""
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return (
        bool(value)
        and value[0] in letters + "_"
        and all(character in letters + "_0123456789" for character in value[1:])
    )


def characterize(
    *,
    hw_source: str | Path,
    module: str,
    inputs: dict[str, int],
    outputs: dict[str, int],
    cases: list[dict],
    circt_opt: str | Path,
    verilator: str | Path,
    output: str | Path,
    reference: dict,
    domain: dict,
) -> dict:
    """Emit source, vectors, native harness and a cell-only differential receipt.

    Ports are explicit raw-bit scalars (1..64 bits). Cases must supply every port;
    a masked output comparison is not allowed. A new output directory is required.
    This interface is deliberately combinational: it supplies no clocks/resets.
    """
    source, circt, simulator = (Path(path).resolve(strict=True) for path in (hw_source, circt_opt, verilator))
    root = Path(output).absolute()
    if root.exists():
        raise ValueError("cell characterization requires a new output directory")
    if not _simple_identifier(module):
        raise ValueError("cell module must have a native C++ identifier")
    for ports in (inputs, outputs):
        if not ports or any(
            not _simple_identifier(name) or type(bits) is not int or not 1 <= bits <= 64 for name, bits in ports.items()
        ):
            raise ValueError("explicit scalar input/output ports must have widths in 1..64")
    if set(inputs) & set(outputs) or not cases or not reference or not domain:
        raise ValueError("cell characterization requires distinct ports, vectors, reference identity and domain")
    for case in cases:
        for field, ports in (("inputs", inputs), ("expected", outputs)):
            if set(case.get(field, {})) != set(ports) or any(
                type(case[field][name]) is not int or not 0 <= case[field][name] < (1 << bits)
                for name, bits in ports.items()
            ):
                raise ValueError("each vector must supply exact unsigned raw bits for every declared port")
    source_bytes = source.read_bytes()
    source_identity = {"path": str(source), "sha256": hashlib.sha256(source_bytes).hexdigest()}
    sliced, included, missing = extract(source_bytes.decode(), module)
    if missing:
        raise ValueError(f"cell closure contains unresolved module references: {missing}")
    # No clock stimulus, latency or reset model is supplied by this interface.
    # Refuse sequential dialect operations instead of silently testing cycle zero.
    if "seq." in sliced or "sv.alwaysff" in sliced:
        raise ValueError("combinational characterization cannot qualify sequential cell operations")
    root.mkdir(parents=True)
    (root / "cell.hw.mlir").write_text(sliced)
    (root / "cases.json").write_text(json.dumps(cases, sort_keys=True) + "\n")
    lines = [
        " ".join(
            str(case[field][name]) for field, ports in (("inputs", inputs), ("expected", outputs)) for name in ports
        )
        for case in cases
    ]
    (root / "vectors.txt").write_text("\n".join(lines) + "\n")
    variables = ", ".join(f"v{index}" for index in range(len(inputs) + len(outputs)))
    reads = " >> ".join(f"v{index}" for index in range(len(inputs) + len(outputs)))
    assignments = "\n".join(f"    dut.{name} = v{index};" for index, name in enumerate(inputs))
    checks = "\n".join(
        f"    if (uint64_t(dut.{name}) != v{index + len(inputs)}) {{ "
        f'if (failed < 16) std::cout << "mismatch " << count << " {name} expected " '
        f'<< v{index + len(inputs)} << " observed " << uint64_t(dut.{name}) << "\\n"; ++failed; }}'
        for index, name in enumerate(outputs)
    )
    (root / "harness.cpp").write_text(
        f'#include "V{module}.h"\n#include <cstdint>\n#include <fstream>\n#include <iostream>\n'
        f"int main(int argc, char** argv) {{\n  if (argc != 2) return 4;\n  std::ifstream data(argv[1]);\n"
        f"  V{module} dut; uint64_t {variables}; unsigned count = 0, failed = 0;\n"
        f"  while (data >> {reads}) {{\n{assignments}\n    dut.eval();\n{checks}\n    ++count;\n  }}\n"
        f'  std::cout << "vectors " << count << " failed_outputs " << failed << "\\n";\n'
        f"  return count != {len(cases)} ? 4 : failed ? 2 : 0;\n}}\n"
    )
    sv = root / "verilog"
    sv.mkdir()
    stages = [
        (
            "lower",
            [
                str(circt),
                str(root / "cell.hw.mlir"),
                "--lower-seq-to-sv",
                "--lower-hw-to-sv",
                "--hw-legalize-modules",
                f"--export-split-verilog=dir-name={sv}",
                "-o",
                str(root / "cell.sv.mlir"),
            ],
        ),
    ]
    receipt = {
        "schema": "merlin.rtl_cell_characterization.v1",
        "module": module,
        "status": "not_executed",
        "qualification_scope": "selected combinational cell and declared vectors only",
        "source": source_identity,
        "tools": {"circt_opt": _identity(circt), "verilator": _identity(simulator)},
        "producer": _identity(Path(__file__)),
        "included_modules": included,
        "inputs": inputs,
        "outputs": outputs,
        "reference": reference,
        "domain": domain,
        "vector_count": len(cases),
        "stages": [],
        "whole_operation_support": "not_qualified",
        "compiler_support": "not_qualified",
    }
    try:
        for name, argv in stages:
            code = stream_stamped(
                argv,
                cwd=root,
                transcript=root / f"{name}.stdout.log",
                stderr_path=root / f"{name}.stderr.log",
                timeout=60,
            )
            receipt["stages"].append({"stage": name, "argv": argv, "returncode": code})
            if code:
                raise RuntimeError(f"native cell {name} failed with return code {code}")
        argv = [
            str(simulator),
            "--cc",
            "--exe",
            "--build",
            "-j",
            "2",
            "-Wno-fatal",
            "--top-module",
            module,
            "--Mdir",
            str(root / "native"),
            *[str(path) for path in sorted(sv.glob("*.sv"))],
            str(root / "harness.cpp"),
        ]
        code = stream_stamped(
            argv, cwd=root, transcript=root / "build.stdout.log", stderr_path=root / "build.stderr.log", timeout=60
        )
        receipt["stages"].append({"stage": "build", "argv": argv, "returncode": code})
        if code:
            raise RuntimeError(f"native cell build failed with return code {code}")
        argv = [str(root / "native" / f"V{module}"), str(root / "vectors.txt")]
        code = stream_stamped(
            argv, cwd=root, transcript=root / "execute.stdout.log", stderr_path=root / "execute.stderr.log", timeout=60
        )
        receipt["stages"].append({"stage": "execute", "argv": argv, "returncode": code})
        receipt["status"] = "passed" if code == 0 else "mismatch" if code == 2 else "failed"
        if _identity(source) != source_identity or any(
            _identity(Path(identity["path"])) != identity for identity in receipt["tools"].values()
        ):
            raise RuntimeError("selected source or native tools changed during characterization")
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    receipt["artifacts"] = {
        path.relative_to(root).as_posix(): _identity(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "native" not in path.relative_to(root).parts
    }
    native = root / "native" / f"V{module}"
    if native.is_file():
        receipt["artifacts"][f"native/V{module}"] = _identity(native)
    (root / "characterization.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def verify_characterization(path: str | Path, *, source_sha256: str | None = None) -> dict:
    """Verify a moved characterization artifact without reopening live RTL/tools.

    A caller must separately bind this receipt's bytes to its own immutable input
    selection. This checks content integrity and observed execution, not signer
    authenticity, domain completeness, or a whole-kernel numerical guarantee.
    """
    selected = Path(path).resolve(strict=True)
    receipt = json.loads(selected.read_bytes())
    if receipt.get("schema") != "merlin.rtl_cell_characterization.v1" or receipt.get("status") != "passed":
        raise ValueError("characterization does not record successful cell execution")
    source = (receipt.get("source") or {}).get("sha256")
    if not is_sha256(source) or (source_sha256 is not None and source != source_sha256):
        raise ValueError("characterization is bound to a different or unknown HW source")
    stages = receipt.get("stages")
    if not isinstance(stages, list) or [(row.get("stage"), row.get("returncode")) for row in stages] != [
        ("lower", 0),
        ("build", 0),
        ("execute", 0),
    ]:
        raise ValueError("characterization did not complete lowering, native build and execution")
    artifacts = receipt.get("artifacts") or {}
    required = {
        "cases.json",
        "vectors.txt",
        "harness.cpp",
        "cell.hw.mlir",
        "execute.stdout.log",
        f"native/V{receipt.get('module')}",
    }
    if not required <= artifacts.keys():
        raise ValueError("characterization omits required source/vector/native artifacts")
    for name, identity in artifacts.items():
        relative = Path(name)
        member = selected.parent / relative
        if relative.is_absolute() or ".." in relative.parts or member.is_symlink() or not member.is_file():
            raise ValueError("characterization contains an unsafe or unavailable artifact")
        if any(
            selected.parent.joinpath(*relative.parts[:index]).is_symlink() for index in range(1, len(relative.parts))
        ):
            raise ValueError("characterization artifact traverses a symlink")
        if hashlib.sha256(member.read_bytes()).hexdigest() != identity.get("sha256"):
            raise ValueError(f"characterization artifact changed: {name}")
    count = receipt.get("vector_count")
    if type(count) is not int or count < 1 or len(json.loads((selected.parent / "cases.json").read_bytes())) != count:
        raise ValueError("characterization vector count differs from its saved cases")
    if (selected.parent / "execute.stdout.log").read_text().splitlines() != [f"vectors {count} failed_outputs 0"]:
        raise ValueError("characterization execution log does not establish zero failed outputs")
    return receipt
