#!/usr/bin/env python3
"""Build the OPU's spike functional model (``--extlib``) from the unit's own RTL facts.

    python spike/build.py --out <dir> [--config KodiakOPUConfig] [--cxx g++]

Run it with a Merlin environment (it uses ``merlin.targetgen.rtl.opu_isa`` through this package's
``opu_shim``, exactly as the compile route does) and ``$MERLIN_CHIPYARD`` pointing at the checkout pinned
as ``saturn_opu_int8``. It writes, under ``--out``:

* ``opu_encodings.h``  -- GENERATED: the four instruction words (match/mask), their mnemonics, the
  vector/datapath geometry of the named config and the datapath parameters, all read from Chisel.
  ``opu_ext.cc`` contains no encoding and no geometry of its own.
* ``lib<extension>_model.so`` -- the extension, compiled against the headers installed beside the
  ``spike`` Merlin runs (the extension resolves its spike symbols from that binary at load time, so it is
  only valid for that spike build; its digest is recorded).
* ``manifest.json`` -- what was read (pin verification, source digests), what was built (sha256 of the
  .so, compiler, spike binary digest), and the encodings and geometry compiled in.

``contracts/overlay.yaml`` declares the model by its INPUTS (``runner.spike_extension.built_from``: the
source, encodings-header and model digests this manifest records), so a build on any machine is accepted
as long as its manifest describes the .so and those inputs and the self-test passed. Changing the model
source or the RTL it derives from changes those digests, which must then be re-declared and re-reviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = HERE / "model.yaml"
SOURCE = HERE / "opu_ext.cc"
HEADER = "opu_encodings.h"

#: The roles the model implements, the operand pattern each must have in the RTL's own `props`, and the
#: funct3 class (instantiation form) it must have been instantiated as. A role whose RTL declaration says
#: otherwise is a different instruction than the one opu_ext.cc models, so the build refuses.
ROLE_SHAPE = {
    "accumulate": ("VV", {"ReadsVS1": True, "ReadsVS2": True, "WritesVD": False}),
    "operand_load": ("VX", {"ReadsVS1": False, "ReadsVS2": True, "WritesVD": False}),
    "broadcast": ("VX", {"ReadsVS1": False, "ReadsVS2": True, "WritesVD": False}),
    "readout": ("VX", {"ReadsVS1": False, "ReadsVS2": False, "WritesVD": True}),
}

# R-type field masks (opcode, funct3, funct6). Format facts of RISC-V, not of this unit; vm (bit 25) is
# left out because the unit's sequencer never reads the mask register.
_OPCODE_MASK = 0x7F
_FUNCT3_SHIFT, _FUNCT3_MASK = 12, 0x7
_FUNCT6_SHIFT, _FUNCT6_MASK = 26, 0x3F


class BuildError(RuntimeError):
    pass


def _load_shim():
    spec = importlib.util.spec_from_file_location("kodiak_opu_shim", ROOT / "opu_shim.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _strip_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        i = line.find("//")
        out.append(line if i < 0 else line[:i])
    return "\n".join(out)


def _balanced(text: str, header: str, opener: str = "(", closer: str = ")") -> str | None:
    i = text.find(header)
    if i < 0:
        return None
    j = text.find(opener, i + len(header) - 1)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == opener:
            depth += 1
        elif text[k] == closer:
            depth -= 1
            if depth == 0:
                return text[j + 1 : k]
    return None


def _split_top(text: str) -> list[str]:
    out, cur, depth = [], [], 0
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return [s.strip() for s in out if s.strip()]


def case_class_defaults(text: str, cls: str) -> dict[str, int]:
    """``{field: default}`` for the integer fields of ``case class <cls> ( val a : Int = 8, ... )``."""
    body = _balanced(_strip_comments(text), f"case class {cls}")
    if body is None:
        return {}
    out: dict[str, int] = {}
    for arg in _split_top(body):
        decl, sep, default = arg.partition("=")
        if not sep:
            continue
        name = decl.strip().removeprefix("val ").partition(":")[0].strip()
        try:
            out[name] = int(default.strip(), 0)
        except ValueError:
            continue
    return out


def instantiation_args(text: str, cls: str) -> list[list[str]]:
    """The argument lists of every ``<cls>(...)`` call (not the declaration) in ``text``."""
    text = _strip_comments(text)
    calls, start = [], 0
    while True:
        i = text.find(cls + "(", start)
        if i < 0:
            return calls
        start = i + len(cls)
        before = text[max(0, i - 12) : i]
        if before.rstrip().endswith("case class") or (i > 0 and (text[i - 1].isalnum() or text[i - 1] == "_")):
            continue
        body = _balanced(text[i:], cls + "(")
        calls.append(_split_top(body or ""))


def derive(config: str | None) -> dict:
    """Everything the generated header carries, each read from the unit's sources."""
    from merlin.common import provenance as PROV
    from merlin.targetgen.rtl import opu_isa

    model = yaml.safe_load(MODEL.read_text(encoding="utf-8"))
    shim = _load_shim()
    contract = shim.load_contract(str(model["unit"]), path=ROOT / "contracts" / "matrix_units.yaml")
    if config is None:
        target_contract = yaml.safe_load((ROOT / "contracts" / "target_contract.yaml").read_text(encoding="utf-8"))
        config = (target_contract.get("rtl_source") or {}).get("config")
        if not config:
            raise BuildError("no --config and the contract declares no rtl_source.config")

    gaps: list[str] = []
    dp = model["datapath"]
    behaviour = [str(p) for p in model.get("behaviour_sources", ())]
    reads = [
        *contract.sources.values(),
        *(str(c) for c in contract.configs.get("config_scala", ())),
        *(str(m) for m in contract.configs.get("mixin_scala", ())),
        str(dp["source"]),
        *behaviour,
    ]
    verification = PROV.verify(contract.pin, checkout=contract.checkout(), reads=reads)
    if not verification.ok:
        gaps.append(f"pin {contract.pin} does not verify: {list(verification.drift)} missing={list(verification.missing_paths)}")

    derived = shim.derive_encodings(contract)
    if not derived.ok:
        raise BuildError(f"encoding derivation is not ok (gaps={list(derived.gaps)}, crosschecks={list(derived.crosschecks)})")

    roles = {}
    for role, (form, flags) in ROLE_SHAPE.items():
        name = contract.kernel_roles.get(role)
        enc = derived.encodings.get(name) if name else None
        if enc is None:
            raise BuildError(f"role {role!r} names {name!r}, which the derivation did not produce")
        got = {k: bool(enc.flags.get(k)) for k in flags}
        if enc.form != form or got != flags:
            raise BuildError(
                f"role {role!r} ({name}) is {enc.form} with {got} in the RTL, but the model implements a "
                f"{form} instruction with {flags}; refusing to compile semantics onto a different instruction"
            )
        match = enc.opcode | (enc.funct3 << _FUNCT3_SHIFT) | (enc.funct6 << _FUNCT6_SHIFT)
        mask = _OPCODE_MASK | (_FUNCT3_MASK << _FUNCT3_SHIFT) | (_FUNCT6_MASK << _FUNCT6_SHIFT)
        roles[role] = {"instruction": name, "match": match, "mask": mask, "opcode": enc.opcode,
                       "funct3": enc.funct3, "funct6": enc.funct6, "funct7": enc.funct7}

    cfg_text = "\n".join(p.read_text(encoding="utf-8") for p in contract.config_scala())
    mixin_text = "\n".join(p.read_text(encoding="utf-8") for p in contract.mixin_scala())
    geom = opu_isa.vector_unit_params(cfg_text, config, mixin_text=mixin_text)
    if "vLen" not in geom or "dLen" not in geom:
        raise BuildError(f"could not read vLen/dLen of {config} (got {geom})")

    dp_path = contract.checkout() / str(dp["source"])
    dp_text = dp_path.read_text(encoding="utf-8")
    defaults = case_class_defaults(dp_text, str(dp["params_class"]))
    params = {}
    for key, field in dp["params_fields"].items():
        if field not in defaults:
            raise BuildError(f"{dp['params_class']}.{field} has no integer default in {dp_path}")
        params[key] = defaults[field]
    lmul_defs = opu_isa.bit_literal_defs(dp_text, str(dp["params_trait"]))
    if str(dp["max_lmul_def"]) not in lmul_defs:
        raise BuildError(f"{dp['params_trait']}.{dp['max_lmul_def']} is not a literal in {dp_path}")
    params["max_lmul"] = lmul_defs[str(dp["max_lmul_def"])]
    # The defaults are only the values if nothing overrides them where the unit is instantiated.
    for label in ("params",):
        overrides = [a for a in instantiation_args(contract.source(label).read_text(encoding="utf-8"),
                                                   str(dp["params_class"])) if a]
        if overrides:
            raise BuildError(f"{dp['params_class']} is instantiated WITH arguments {overrides} in "
                             f"{contract.source(label)}; the defaults are not the elaborated values")

    sources = sorted({contract.checkout() / r for r in reads if (contract.checkout() / r).is_file()} |
                     set(contract.config_scala()))
    return {
        "unit": contract.unit,
        "extension_name": str(model["extension_name"]),
        "config": config,
        "geometry": {"vlen": int(geom["vLen"]), "dlen": int(geom["dLen"])},
        "datapath": params,
        "roles": roles,
        "pin": contract.pin,
        "pin_ok": verification.ok,
        "gaps": gaps,
        "provenance": PROV.record(pins={contract.pin: verification}, sources=sources),
        "source_digests": {str(p): _sha256(p) for p in sources},
    }


def render_header(facts: dict) -> str:
    g, d = facts["geometry"], facts["datapath"]
    lines = [
        "/* GENERATED by kodiak-mlir spike/build.py from the unit's Chisel -- do not edit.",
        f" * unit {facts['unit']}, config {facts['config']}, pin {facts['pin']} (verified: {facts['pin_ok']}).",
        " * Encodings: merlin.targetgen.rtl.opu_isa, cross-checked against the unit's bme.h. */",
        "#pragma once",
        f"#define OPU_EXT_NAME {facts['extension_name']}",
        f'#define OPU_EXT_NAME_STR "{facts["extension_name"]}"',
        f'#define OPU_CONFIG "{facts["config"]}"',
        f"#define OPU_VLEN {g['vlen']}",
        f"#define OPU_DLEN {g['dlen']}",
        f"#define OPU_A_BITS {d['a_bits']}",
        f"#define OPU_B_BITS {d['b_bits']}",
        f"#define OPU_C_BITS {d['c_bits']}",
        f"#define OPU_MATRIX_REGS {d['matrix_regs']}",
        f"#define OPU_MAX_LMUL {d['max_lmul']}",
    ]
    for role, r in facts["roles"].items():
        key = role.upper()
        lines += [
            f"/* {role}: {r['instruction']} opcode={r['opcode']:#04x} funct3={r['funct3']} funct6={r['funct6']} */",
            f"#define OPU_{key}_MATCH {r['match']:#010x}u",
            f"#define OPU_{key}_MASK {r['mask']:#010x}u",
            f'#define OPU_{key}_MNEMONIC "{r["instruction"].lower()}"',
        ]
    return "\n".join(lines) + "\n"


def spike_layout() -> tuple[Path, Path]:
    """``(spike binary, its installed include dir)`` -- the spike Merlin's runners use."""
    from merlin.runtime.backends import spike as spike_backend

    spike = Path(spike_backend.spike_path())
    inc = spike.parent.parent / "include"
    if not (inc / "riscv" / "extension.h").is_file():
        raise BuildError(f"{inc} has no riscv/extension.h; the extension must be built against {spike}'s headers")
    return spike, inc


def softfloat_types(inc: Path) -> Path:
    """``softfloat_types.h``, which the installed spike headers include but do not install."""
    cand = [inc / "softfloat" / "softfloat_types.h"]
    cy = os.environ.get("MERLIN_CHIPYARD")
    if cy:
        cand.append(Path(cy) / "toolchains" / "riscv-tools" / "riscv-isa-sim" / "softfloat" / "softfloat_types.h")
    for c in cand:
        if c.is_file():
            return c
    raise BuildError(f"softfloat_types.h not found (looked in {[str(c) for c in cand]})")


def build(out: Path, *, config: str | None, cxx: str) -> dict:
    facts = derive(config)
    out.mkdir(parents=True, exist_ok=True)
    header = out / HEADER
    header.write_text(render_header(facts), encoding="utf-8")

    spike, inc = spike_layout()
    # A self-contained include tree: the installed headers reach "../softfloat/" and "../fesvr/".
    stage = out / "include"
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(inc / "riscv", stage / "riscv")
    shutil.copytree(inc / "fesvr", stage / "fesvr")
    sf = softfloat_types(inc)
    (stage / "softfloat").mkdir()
    shutil.copy2(sf, stage / "softfloat" / "softfloat_types.h")

    so = out / f"lib{facts['extension_name']}_model.so"
    cmd = [cxx, "-std=c++17", "-O2", "-fPIC", "-shared", "-Wall", "-Wno-unused-function",
           "-I", str(out), "-I", str(stage / "riscv"), "-I", str(stage),
           str(SOURCE), "-o", str(so)]
    got = subprocess.run(cmd, capture_output=True, text=True)
    if got.returncode != 0:
        raise BuildError(f"compile failed:\n{' '.join(cmd)}\n{got.stderr[-4000:]}")
    cxx_version = subprocess.run([cxx, "--version"], capture_output=True, text=True).stdout.splitlines()[:1]
    manifest = {
        "schema": "kodiak.spike_extension.v1",
        "extension_name": facts["extension_name"],
        "extlib": str(so),
        "sha256": _sha256(so),
        "header_sha256": _sha256(header),
        "source_sha256": _sha256(SOURCE),
        "model_sha256": _sha256(MODEL),
        "spike": {"binary": str(spike), "sha256": _sha256(spike), "include": str(inc),
                  "softfloat_types": str(sf), "softfloat_types_sha256": _sha256(sf)},
        "compiler": {"cxx": cxx, "version": cxx_version, "command": cmd},
        **{k: facts[k] for k in ("unit", "config", "geometry", "datapath", "roles", "pin", "pin_ok", "gaps",
                                 "source_digests", "provenance")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return manifest


def selftest(out: Path, manifest: dict, *, timeout: int = 600) -> dict:
    """Build spike/selftest.c against the generated header, run it on spike with the model, parse it.

    Covers the forms the certification corpus does not (move-in, column forms, every tile register,
    vl == 0, readout into v16+). The corpus itself is run by Merlin (merlin.kernels.opu_cert)."""
    from merlin.runtime.backends import spike as spike_backend

    harness = spike_backend.harness_dir()
    elf = out / "selftest.elf"
    cmd = [str(spike_backend.gcc_path()), "-march=rv64gcv", "-mabi=lp64d", "-mcmodel=medany", "-O2",
           "-ffreestanding", "-nostdlib", "-nostartfiles", "-I", str(out), "-I", str(harness),
           "-T", str(harness / "link.ld"),
           *(str(harness / f) for f in ("crt.S", "htif.c", "printf_min.c")),
           str(HERE / "selftest.c"), "-o", str(elf)]
    got = subprocess.run(cmd, capture_output=True, text=True)
    if got.returncode != 0:
        raise BuildError(f"selftest did not compile:\n{got.stderr[-3000:]}")
    vlen = manifest["geometry"]["vlen"]
    run = [str(spike_backend.spike_path()), f"--isa=rv64gcv_zvl{vlen}b", f"--extlib={manifest['extlib']}",
           f"--extension={manifest['extension_name']}", str(elf)]
    proc = subprocess.run(run, capture_output=True, text=True, timeout=timeout)
    checks, done = {}, None
    for line in (proc.stdout + proc.stderr).splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "SELFTEST":
            checks[parts[1]] = int(parts[2])
        elif len(parts) == 2 and parts[0] == "SELFTEST_DONE":
            done = int(parts[1])
    ok = done == 0 and bool(checks) and all(v == 0 for v in checks.values())
    return {"ok": ok, "checks": checks, "returncode": proc.returncode,
            "console_tail": (proc.stdout + proc.stderr)[-2000:] if not ok else ""}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path, help="build directory (e.g. <merlin>/out/build/...)")
    ap.add_argument("--config", help="elaborated config (default: the contract's rtl_source.config)")
    ap.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    ap.add_argument("--no-selftest", action="store_true", help="skip running spike/selftest.c on the model")
    args = ap.parse_args(argv)
    out = args.out.resolve()
    try:
        m = build(out, config=args.config, cxx=args.cxx)
        if not args.no_selftest:
            m["selftest"] = selftest(out, m)
            (out / "manifest.json").write_text(json.dumps(m, indent=2, default=str) + "\n", encoding="utf-8")
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if "selftest" in m:
        print(f"selftest: {'ok' if m['selftest']['ok'] else 'FAILED'} {m['selftest']['checks']}")
        if not m["selftest"]["ok"]:
            print(m["selftest"]["console_tail"], file=sys.stderr)
            return 1
    for gap in m["gaps"]:
        print(f"warning: {gap}", file=sys.stderr)
    print(f"built {m['extlib']}")
    print(f"  config {m['config']}: vlen={m['geometry']['vlen']} dlen={m['geometry']['dlen']} {m['datapath']}")
    print(f"  sha256 {m['sha256']}")
    print(f"  spike --extlib={m['extlib']} --extension={m['extension_name']} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
