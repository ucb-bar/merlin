#!/usr/bin/env python3
"""Run your submission's declared commands on one case, the way the grader does, and keep every stage.

  python dump_stages.py run CASE [--submission submission|variants/NAME] [--out DIR] [--append ARG]...
  python dump_stages.py compare DIR_A DIR_B
  python dump_stages.py check DIR [--ws .]

``run`` reads ``manifest.yaml`` in the selected tree, resolves the tool (``build.tool_output`` when a
build block is declared, else ``entrypoints.tool``), substitutes only ``{tool}``, ``{input_mlir}`` and
``{output_json}`` with absolute paths, and runs each command from the package root:
``parse``; ``lower_interface_to_target`` (stdout kept as ``lowered.target.mlir``); then
``emit_analysis_bundle`` if declared, else ``emit_command_buffer`` and ``lower_target_to_llvm`` (or
its alias ``emit_target_artifact``), keeping ``command_buffer.json`` and ``lowered.llvm.mlir``. Every
stage's argv, exit code, time and stderr go to ``stages.json`` and ``STAGE.stderr``. ``--append``
adds your tool's own debug flags to every command. CASE is a ``capsule.interface.mlir``, a case
directory, or a case name found as ``*/CASE/capsule.interface.mlir`` under the workspace.

``compare`` names the first stage whose output differs between two dumps and prints a short diff.
``check`` applies the observable contracts: interface module attributes and op names against
``interface_dialect_contract.yaml``; non-empty target IR (parsed with xDSL, unregistered ops allowed,
when xDSL is importable); a JSON command buffer whose declined entries carry a reason; a lowered
module with a function. Standard library plus optional PyYAML/xDSL. Never writes into the tree it runs.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PLACEHOLDERS = ("{tool}", "{input_mlir}", "{output_json}")
ALIASES = {"emit_target_artifact": "lower_target_to_llvm", "lower_target_to_llvm": "emit_target_artifact"}
#: Stage outputs in pipeline order; ``compare`` reports the first that differs.
OUTPUTS = ("input.interface.mlir", "lowered.target.mlir", "command_buffer.json", "lowered.llvm.mlir")
CASE_FILE = "capsule.interface.mlir"


def _yaml(path: Path):
    try:
        import yaml
    except ImportError as exc:  # the author's interpreter normally has it
        raise SystemExit(f"reading {path.name} needs PyYAML: {exc}") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def find_case(ws: Path, case: str) -> Path:
    path = Path(case)
    if path.is_file():
        return path.resolve()
    if path.is_dir() and (path / CASE_FILE).is_file():
        return (path / CASE_FILE).resolve()
    hits = sorted(p for p in ws.glob(f"*/{case}/{CASE_FILE}") if "selfcheck_out" not in p.parts)
    if len(hits) != 1:
        raise SystemExit(f"case {case!r}: expected one */{case}/{CASE_FILE} under {ws}, found {len(hits)}")
    return hits[0].resolve()


def resolve_argv(template: list, *, tool: Path, input_mlir: Path, output_json: Path | None) -> list[str]:
    """The grader's substitution: three placeholders, absolute paths, any other ``{...}`` refused."""
    out = []
    for token in template:
        token = str(token).replace("{tool}", str(tool)).replace("{input_mlir}", str(input_mlir))
        if output_json is not None:
            token = token.replace("{output_json}", str(output_json))
        lo = token.find("{")
        if lo != -1 and token.find("}", lo) != -1:
            raise SystemExit(f"unsubstituted placeholder in {token!r}; only {', '.join(PLACEHOLDERS)} exist")
        out.append(token)
    return out


def _needs_interpreter(argv: list[str], tool: Path, language: str) -> bool:
    if not argv or argv[0] != str(tool) or not tool.is_file() or tool.stat().st_mode & 0o111:
        return False
    if language.lower() == "python" or tool.suffix == ".py":
        return True
    with tool.open("rb") as stream:
        first = stream.readline(128)
    return first.startswith(b"#!") and b"python" in first


def plan(manifest: dict) -> list[tuple[str, str | None]]:
    """``(command, stdout file)`` in grader order for this manifest."""
    commands = manifest.get("commands") or {}
    steps: list[tuple[str, str | None]] = [("parse", None), ("lower_interface_to_target", "lowered.target.mlir")]
    if "emit_analysis_bundle" in commands:
        return [*steps, ("emit_analysis_bundle", "lowered.llvm.mlir")]
    artifact = "lower_target_to_llvm" if "lower_target_to_llvm" in commands else ALIASES["lower_target_to_llvm"]
    return [*steps, ("emit_command_buffer", None), (artifact, "lowered.llvm.mlir")]


def run(args) -> int:
    ws = Path(args.ws).resolve()
    pkg = (ws / args.submission).resolve()
    manifest = _yaml(pkg / "manifest.yaml")
    build = manifest.get("build")
    tool = (pkg / (build["tool_output"] if build else manifest["entrypoints"]["tool"])).resolve()
    if build and not tool.exists():
        print(f"note: manifest declares a build block; {tool.name} is not built yet", file=sys.stderr)
    source = find_case(ws, args.case)
    out = Path(args.out or Path(os.environ.get("TMPDIR") or ws) / "ir-stages" / source.parent.name).resolve()
    if out.is_relative_to(pkg):
        raise SystemExit("--out must be outside the tree being run (its digest would change)")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, out / "input.interface.mlir")
    inp, cb = out / "input.interface.mlir", out / "command_buffer.json"
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    only = {s for s in (args.only or "").split(",") if s}
    record = {"submission": args.submission, "case": str(source), "tool": str(tool), "stages": []}
    status = 0
    for name, stdout_name in plan(manifest):
        if only and name not in only:
            continue
        template = (manifest.get("commands") or {}).get(name, {}).get("argv")
        if template is None:
            record["stages"].append({"stage": name, "skipped": "not declared"})
            continue
        argv = resolve_argv(template, tool=tool, input_mlir=inp, output_json=cb) + list(args.append or ())
        if _needs_interpreter(argv, tool, str(manifest.get("language", ""))):
            argv = [sys.executable, *argv]
        started = time.monotonic()
        try:
            proc = subprocess.run(argv, cwd=pkg, env=env, capture_output=True, text=True, timeout=args.timeout,
                                  check=False)
            rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            rc, stdout, stderr = "timeout", exc.stdout or "", f"timed out after {args.timeout}s"
        row = {"stage": name, "argv": argv, "returncode": rc, "seconds": round(time.monotonic() - started, 3)}
        (out / f"{name}.stderr").write_text(stderr if isinstance(stderr, str) else stderr.decode(), encoding="utf-8")
        if stdout_name:
            (out / stdout_name).write_text(stdout if isinstance(stdout, str) else stdout.decode(), encoding="utf-8")
            row["stdout"] = stdout_name
        if rc != 0 or (stdout_name and not str(stdout).strip()):
            row["failed"] = True
            status = 1
        record["stages"].append(row)
        print(f"{name:<28} rc={rc!s:<7} {row['seconds']:>7.2f}s  {stdout_name or ''}")
        if row.get("failed"):
            print(f"  stderr tail: {str(stderr)[-600:]}")
            break
    record["outputs"] = {n: _digest(out / n) for n in OUTPUTS if (out / n).is_file()}
    (out / "stages.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"stages in {out}")
    return status


def _digest(path: Path) -> dict:
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def compare(args) -> int:
    a, b = Path(args.a), Path(args.b)
    for name in OUTPUTS:
        pa, pb = a / name, b / name
        if not pa.is_file() or not pb.is_file():
            print(f"{name}: present in {'a' if pa.is_file() else 'neither' if not pb.is_file() else 'b'} only")
            return 1
        if pa.read_bytes() != pb.read_bytes():
            print(f"first differing stage output: {name}")
            lines = difflib.unified_diff(pa.read_text(encoding="utf-8").splitlines(),
                                         pb.read_text(encoding="utf-8").splitlines(), str(pa), str(pb), lineterm="", n=2)
            for i, line in enumerate(lines):
                if i >= args.lines:
                    print("... (diff truncated; raise --lines)")
                    break
                print(line)
            return 1
    print("every stage output is byte-identical")
    return 0


def _module_attributes(text: str) -> dict:
    head = text.partition("attributes {")[2].partition("}")[0]
    attrs = {}
    for item in head.split(","):
        key, eq, value = item.partition("=")
        if eq:
            attrs[key.strip()] = value.strip().strip('"')
    return attrs


def op_names(text: str) -> dict:
    counts: dict = {}
    for line in text.splitlines():
        body = line.strip()
        if not body or body.startswith(("^", "}", "//", "#", "module", "builtin.module")):
            continue
        head, eq, rest = body.partition(" = ")
        if eq and head.startswith("%"):
            body = rest
        token = body.split(" ", 1)[0].split("(", 1)[0].strip('"')
        if "." in token and token[:1].isalpha():
            counts[token] = counts.get(token, 0) + 1
    return counts


def check(args) -> int:
    ws, d = Path(args.ws).resolve(), Path(args.dir)
    findings, notes = [], []
    iface = d / "input.interface.mlir"
    contract_path = ws / "interface_dialect_contract.yaml"
    if iface.is_file() and contract_path.is_file():
        contract = _yaml(contract_path).get("dialect") or {}
        text = iface.read_text(encoding="utf-8")
        attrs = _module_attributes(text)
        for spec in contract.get("module_attributes") or ():
            name = spec.get("name")
            if name not in attrs:
                findings.append(f"interface: module attribute {name} missing")
            elif "must_equal" in spec and attrs[name] != str(spec["must_equal"]):
                findings.append(f"interface: {name} = {attrs[name]!r}, contract requires {spec['must_equal']!r}")
        declared = {op.get("name") for key, ops in contract.items() if key.endswith("_ops") and isinstance(ops, list)
                    for op in ops if isinstance(op, dict)}
        namespace = str(contract.get("namespace") or "")
        used = {op for op in op_names(text) if op.startswith(namespace + ".")}
        if used - declared:
            findings.append(f"interface: ops not in the contract: {sorted(used - declared)}")
        notes.append(f"interface ops used: {sorted(used)}")
    target = d / "lowered.target.mlir"
    if target.is_file():
        text = target.read_text(encoding="utf-8")
        if not text.strip():
            findings.append("target IR is empty (the contract requires non-empty, parseable MLIR)")
        else:
            notes.append(f"target IR dialects: {sorted({op.partition('.')[0] for op in op_names(text)})}")
            try:
                from xdsl.context import Context
                from xdsl.dialects.builtin import Builtin
                from xdsl.parser import Parser
            except ImportError:
                notes.append("xDSL not importable here: target IR parse not checked")
            else:
                context = Context(allow_unregistered=True)
                context.load_dialect(Builtin)
                try:  # upstream dialects (llvm, func, arith, ...) on demand; your own ops stay unregistered
                    from xdsl.dialects import get_all_dialects

                    for name, factory in get_all_dialects().items():
                        if name != "builtin":
                            context.register_dialect(name, factory)
                except (ImportError, ValueError):
                    pass
                try:
                    Parser(context, text).parse_module()
                except Exception as exc:  # noqa: BLE001 -- any parse failure is the finding
                    findings.append(f"target IR does not parse: {str(exc)[:400]}")
    cb = d / "command_buffer.json"
    if cb.is_file():
        try:
            buffer = json.loads(cb.read_text(encoding="utf-8"))
        except ValueError as exc:
            findings.append(f"command buffer is not JSON: {exc}")
        else:
            declined = buffer.get("declined") if isinstance(buffer, dict) else None
            if declined and not (isinstance(declined, dict) and str(declined.get("reason") or "").strip()):
                findings.append("command buffer declares `declined` without a non-empty `reason`")
            elif declined:
                notes.append(f"declined: {declined.get('reason')}")
            notes.append(f"commands: {len(buffer.get('commands') or ()) if isinstance(buffer, dict) else '?'}")
    lowered = d / "lowered.llvm.mlir"
    if lowered.is_file():
        ops = op_names(lowered.read_text(encoding="utf-8"))
        if not any(op.endswith(".func") for op in ops):
            findings.append("lowered module defines no function")
        notes.append(f"lowered ops: {sum(ops.values())}, inline asm sites: "
                     f"{sum(n for op, n in ops.items() if op.endswith('inline_asm'))}")
    print(json.dumps({"findings": findings, "notes": notes}, indent=2))
    return 1 if findings else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("run", help="run the declared commands on one case and keep each stage")
    rp.add_argument("case")
    rp.add_argument("--ws", default=".", help="workspace root (default: current directory)")
    rp.add_argument("--submission", default="submission", help="submission or variants/NAME")
    rp.add_argument("--out", default="", help="output directory (default: $TMPDIR/ir-stages/CASE)")
    rp.add_argument("--append", action="append", help="extra argument appended to every command (repeatable)")
    rp.add_argument("--only", default="", help="comma-separated command names to run")
    rp.add_argument("--timeout", type=int, default=600)
    cp = sub.add_parser("compare", help="first stage whose output differs between two dumps")
    cp.add_argument("a")
    cp.add_argument("b")
    cp.add_argument("--lines", type=int, default=60)
    kp = sub.add_parser("check", help="check one dump against the observable contracts")
    kp.add_argument("dir")
    kp.add_argument("--ws", default=".", help="workspace root holding the contract files")
    args = ap.parse_args(argv)
    return {"run": run, "compare": compare, "check": check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
