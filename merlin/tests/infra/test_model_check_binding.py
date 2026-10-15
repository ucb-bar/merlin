"""The binding receipt binds Spike to the RTL and the FPGA build, names every mismatch and fails closed.

Each test builds a small, self-consistent world on disk (a source checkout with headers, a FIRRTL, a DTS
and CDE configuration sources; a target contract; RTL facts; a board catalog; a binding spec) and then
breaks one binding at a time. The toolchain and Spike are replaced by a fake :class:`Engine`.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest
import yaml

from merlin.model_check import binding as BD
from merlin.model_check import binding_sources as S
from merlin.model_check import bundle as B
from merlin.model_check import cli
from merlin.model_check import report as R
from merlin.model_check.models import Model

TARGET = "fixture_target"
ROOT = "binding_fixture_checkout"
GIB = 1 << 30
BASE = 0x80000000


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _elf(path: Path, segments: list[tuple[int, int]]) -> Path:
    """A minimal little-endian ELF64 with one PT_LOAD per ``(address, size)``."""
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<Q", header, 32, 64)
    struct.pack_into("<HH", header, 54, 56, len(segments))
    table = b"".join(struct.pack("<IIQQQQQQ", 1, 5, 0, a, a, 0, n, 8) for a, n in segments)
    path.write_bytes(bytes(header) + table)
    return path


HEADER = """#ifndef P_H
#define P_H
#include <stdint.h>
#define DIM {dim}
#define BANK_ROWS 4096
#define LONG_MACRO(x) \\
    ((x) + 1)
typedef int8_t elem_t;
typedef int32_t acc_t;
typedef float acc_scale_t;
typedef uint32_t acc_scale_t_bits;
#define ACC_SCALE_EXP_BITS 8 // the exponent
/* a block comment with #define FAKE 1 */
#endif
"""

DTS = """/dts-v1/;

/ {{
\t#address-cells = <1>;
\t#size-cells = <1>;
\tL2: cpus {{
\t\tL13: cpu@0 {{
\t\t\tdevice_type = "cpu";
\t\t\triscv,isa = "rv64imafdc_zicsr_xvendor";
\t\t\ttlb-split;
\t\t}};
\t}};
\tL24: memory@8000000 {{
\t\tdevice_type = "memory";
\t\treg = <0x8000000 0x10000>;
\t\tstatus = "disabled";
\t}};
\tL21: memory@80000000 {{
\t\tdevice_type = "memory";
\t\treg = <0x80000000 {elab_bytes}>;
\t}};
}};
"""

MEM_SCALA = """package mem

import base.Config

class WithDefaultMemPort extends Config((site, here, up) => {
  case ExtMem => Some(MemoryPortParams(MasterPortParams(
                      base = x"8000_0000",
                      size = x"1000_0000",
                      beatBytes = 8), 1))
})

class WithExtMemSize(n: BigInt) extends Config((site, here, up) => {
  case ExtMem => up(ExtMem).map(x => x.copy(master = x.master.copy(size = n)))
})

class WithNChannels(n: Int = 1) extends Config((site, here, up) => {
  case ExtMem => up(ExtMem).map(_.copy(nMemoryChannels = n))
})

class WithNoMemPort extends Config((site, here, up) => {
  case ExtMem => None
})

class BaseConfig extends Config(new WithDefaultMemPort)
"""

SOC_SCALA = """package soc

import mem._

class ElabConfig extends Config(new WithNChannels(1) ++ new BaseConfig)

class FpgaTweaks extends Config(
  // a comment between terms
  new mem.WithExtMemSize((1 << 30) * {gib}L) ++ /* and a block one */
  new Config((_, _, _) => {{ case Unrelated => 3 }})
)

class FpgaConfig extends Config(new FpgaTweaks ++ new ElabConfig)

class Broken extends Config(new WithNoMemPort ++ new ElabConfig)
"""

HOST_SCALA = """package host

class PlatformBase extends Config(new Config((site, here, up) => {
  case HostMemChannelKey => HostMemChannelParams(size = 0x400000000L, // 16 GiB
                                                 beatBytes = 8)
  case HostMemNumChannels => 4
}))

class PlatformConfig extends Config(new Config((_, _, _) => { case HostMemNumChannels => 1 }) ++ new PlatformBase)
"""


class FakeEngine:
    """The toolchain and Spike, as data: ISA strings split into the tokens a toolchain would record."""

    def __init__(self, tmp: Path, *, spike_isa: str, memory: str, image: list[tuple[int, int]] | None = None):
        self.tmp, self.spike_isa, self.memory = tmp, spike_isa, memory
        self.segments = image or [(BASE, 0x2000)]

    def engine(self) -> BD.Engine:
        return BD.Engine(
            canonical_isa=lambda isa: isa.split("_"),
            image_arch=lambda elf: ["rv64i", "m", "a", "f", "d", "c", "zicsr"],
            spike_argv=lambda elf: ["spike", "--extension=fixture", f"--isa={self.spike_isa}", self.memory, str(elf)],
            run=lambda argv: (0, "MCCASE\nMCDONE\n"),
            probe_image=lambda: _elf(self.tmp / "probe.elf", self.segments),
            march="rv64imafdc_zicsr",
        )


def world(
    tmp: Path,
    monkeypatch,
    *,
    spike_dim: int = 16,
    elab_dim: int = 16,
    facts_rows: int = 4096,
    fpga_gib: int = 16,
    board_gib: int = 16,
    elab_dts_bytes: int = 0x10000000,
    elaboration_header: bool = True,
    cite_line: str = "def x(args: Any*): BigInt",
) -> dict:
    checkout = tmp / "checkout"
    for sub in ("spike", "elab", "gen", "bin", "lib", "scala"):
        (checkout / sub).mkdir(parents=True, exist_ok=True)
    (checkout / "spike/params.h").write_text(HEADER.format(dim=spike_dim))
    if elaboration_header:
        (checkout / "elab/params.h").write_text(HEADER.format(dim=elab_dim))
    (checkout / "gen/harness.fir").write_text("FIRRTL version 4.0.0\ncircuit TestHarness :\n")
    (checkout / "gen/harness.dts").write_text(DTS.format(elab_bytes=hex(elab_dts_bytes)))
    (checkout / "bin/spike").write_bytes(b"spike-binary")
    (checkout / "lib/libext.so").write_bytes(b"extension-library")
    (checkout / "spike/libext.so").write_bytes(b"extension-library")
    (checkout / "scala/Mem.scala").write_text(MEM_SCALA)
    (checkout / "scala/Soc.scala").write_text(SOC_SCALA.format(gib=fpga_gib))
    (checkout / "scala/Host.scala").write_text(HOST_SCALA)
    (checkout / "cited.scala").write_text("package object util {\n  def x(args: Any*): BigInt = ???\n}\n")
    (checkout / "pins.yaml").write_text(
        yaml.safe_dump(
            {
                "artifacts": {
                    "bit": {"config": "FpgaConfig", "path": "r/cl_plat-fs-FireSim-FpgaConfig-PlatformConfig/x.tgz"}
                }
            }
        )
    )
    monkeypatch.setenv(f"MERLIN_EXT_{ROOT.upper()}", str(checkout))

    fir = checkout / "gen/harness.fir"
    contract = {
        "name": TARGET,
        "rtl_source": {"ext_root": ROOT, "config": "ElabConfig", "artifact_sha256": {"fir": _sha(fir)}},
        "runner": {
            "backend": "be",
            "be": {
                "engines": {
                    "spike": {
                        "binary": {"path": str(checkout / "bin/spike"), "sha256": _sha(checkout / "bin/spike")},
                        "argv": ["--isa=rv64imafdc", "{elf}"],
                    }
                },
                "readout": {"scalar_abi": {"accumulator_dtype": "i32", "output_dtype": "i8", "scale_dtype": "f32"}},
            },
            "spike_extension": {"extlib": str(checkout / "lib/libext.so"), "sha256": _sha(checkout / "lib/libext.so")},
        },
    }
    contract_path = tmp / "contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract))
    facts = {
        "inputs": {"fir_path": str(fir), "fir_sha256": _sha(fir)},
        "facts": {
            "source": {"config": "ElabConfig"},
            "arrays": [{"name": "mesh", "rows": 16, "cols": 16}],
            "memories": [{"name": "spad", "banks": 4, "depth": facts_rows, "elem_bits": 8}],
            "storage_datapaths": [{"name": "input", "dtype": "i8"}, {"name": "accumulator", "dtype": "i32"}],
        },
    }
    facts_path = tmp / "facts.json"
    facts_path.write_text(json.dumps(facts))
    boards = {
        "schema_version": 1,
        "boards": {
            name: {
                "target": TARGET,
                "host_dts_sha256": _sha(checkout / "gen/harness.dts"),
                "dram_bytes": f"{gib} GiB",
                "dram_base": BASE,
                "harts": 1,
                "console": "htif",
                "flow": "baremetal",
                "code_reserve": "64 MiB",
            }
            for name, gib in (("functional", board_gib), ("fpga_board", 16))
        },
    }
    (checkout / "boards.yaml").write_text(yaml.safe_dump(boards))
    spec = {
        "schema": BD.SPEC_SCHEMA,
        "target": TARGET,
        "board_catalog": "boards.yaml",
        "board": "functional",
        "spike": {"params_header": "spike/params.h", "extension_build": "spike/libext.so"},
        "elaboration": {
            "directory": "gen",
            "firrtl": "harness.fir",
            "dts": "harness.dts",
            "params_header": "elab/params.h",
        },
        "fpga": {
            "pin": {"file": "pins.yaml", "key": ["artifacts", "bit"]},
            "board": "fpga_board",
            "config": "FpgaConfig",
            "platform_config": "PlatformConfig",
            "sources": {"roots": ["scala"], "config_class": "Config", "interpolators": {"x": 16}},
            "target_memory": {"key": "ExtMem", "base": "base", "bytes": "size"},
            "host_memory": {
                "channel_key": "HostMemChannelKey",
                "channel_bytes": "size",
                "channels_key": "HostMemNumChannels",
            },
            "dts": None,
        },
        "citations": [{"claim": "x is hex", "file": "cited.scala", "line": 2, "contains": cite_line}],
        "params": {
            "DIM": {
                "header": {"define": "DIM"},
                "rtl_facts": [
                    {"collection": "arrays", "name": "mesh", "field": "rows"},
                    {"collection": "arrays", "name": "mesh", "field": "cols"},
                ],
            },
            "BANK_ROWS": {
                "header": {"define": "BANK_ROWS"},
                "rtl_facts": [{"collection": "memories", "name": "spad", "field": "depth"}],
            },
            "elem_t": {
                "header": {"typedef": "elem_t"},
                "rtl_facts": [
                    {"collection": "storage_datapaths", "name": "input", "field": "dtype"},
                    {"collection": "memories", "name": "spad", "field": "elem_bits", "as": "bits"},
                ],
                "contract": [{"path": ["runner", "be", "readout", "scalar_abi", "output_dtype"]}],
            },
            "acc_scale_t": {
                "header": {"typedef": "acc_scale_t", "carrier": "acc_scale_t_bits", "exp_bits": "ACC_SCALE_EXP_BITS"},
                "rtl_facts": [
                    {"collection": "interfaces", "name": "layouts", "field": "acc_scale.width", "as": "bits"}
                ],
                "contract": [{"path": ["runner", "be", "readout", "scalar_abi", "scale_dtype"]}],
            },
        },
    }
    spec_path = tmp / "binding.yaml"
    spec_path.write_text(yaml.safe_dump(spec))
    return {"contract_path": contract_path, "facts_path": facts_path, "spec_path": spec_path, "checkout": checkout}


def make(tmp: Path, w: dict, *, spike_isa="rv64imafdc_zicsr", memory=f"-m{hex(BASE)}:{hex(16 * GIB)}", **kw) -> dict:
    fake = FakeEngine(tmp, spike_isa=spike_isa, memory=memory, image=kw.pop("image", None))
    return BD.receipt(
        target=TARGET,
        contract_path=w["contract_path"],
        facts_path=w["facts_path"],
        spec_path=w["spec_path"],
        board=None,
        host_package=None,
        dram_base=BASE,
        dram_bytes=16 * GIB,
        work=tmp / "work",
        engine=fake.engine(),
        **kw,
    )


def by_name(doc: dict) -> dict[str, dict]:
    return {c["name"]: c for c in doc["checks"]}


# ------------------------------------------------------------------------------------------- receipts


def test_a_consistent_world_binds(tmp_path, monkeypatch):
    doc = make(tmp_path, world(tmp_path, monkeypatch))
    checks = by_name(doc)
    assert doc["status"] == "pass", [c for c in doc["checks"] if c["status"] != "pass"]
    assert doc["mismatches"] == [] and doc["unknown"] == []
    assert checks["fpga.target_memory"]["values"]["target_memory"]["bytes"] == 16 * GIB
    assert checks["fpga.host_capacity"]["values"]["channels"] == 1  # the higher-priority fragment wins
    assert any("WithExtMemSize" in site for site in checks["fpga.target_memory"]["values"]["size_from"])
    # acc_scale_t: the extraction records no width, the elaboration header and the contract state it
    assert checks["param.acc_scale_t"]["values"]["not_recorded"] == ["rtl_facts.interfaces[layouts].acc_scale.width"]
    for key in ("spike_binary", "spike_extension", "spike_params_header", "elaboration_params_header", "firrtl", "dts"):
        assert len(doc["inputs"][key]["sha256"]) == 64
    assert BD.digest(doc) == BD.digest(json.loads(BD.serialize(doc)))


def test_every_mismatch_is_named(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch, spike_dim=32, facts_rows=2048, fpga_gib=8)
    doc = make(tmp_path, w, spike_isa="rv64imafdc_zicntr_zicsr")
    failed = {c["name"] for c in doc["checks"] if c["status"] == "fail"}
    assert doc["status"] == "fail"
    assert {
        "params_header_equal",
        "param.DIM",
        "param.BANK_ROWS",
        "isa.spike[0]",
        "dram.fpga_vs_board",
        "dram.fpga_board_entry",
    } <= failed
    checks = by_name(doc)
    assert "DIM" in checks["params_header_equal"]["detail"]
    assert checks["isa.spike[0]"]["values"]["missing"] == ["zicntr"]
    assert hex(8 * GIB) in checks["dram.fpga_vs_board"]["detail"]
    assert all(any(m.startswith(name) for m in doc["mismatches"]) for name in failed)


def test_spike_memory_must_be_the_board_span(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch)
    doc = make(tmp_path, w, memory=f"-m{hex(BASE)}:{hex(2 * GIB)}")
    assert by_name(doc)["dram.spike_vs_board"]["status"] == "fail"
    bare = make(tmp_path, w, memory="-m2048")
    assert by_name(bare)["dram.spike_vs_board"]["status"] == "unknown"


def test_an_image_outside_the_memory_fails_its_footprint(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch)
    doc = make(tmp_path, w, image=[(BASE, 0x1000), (BASE + 17 * GIB, 0x1000)])
    assert by_name(doc)["footprint[0]"]["status"] == "fail"


def test_a_missing_source_is_unknown_and_fails_closed(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch, elaboration_header=False)
    (w["checkout"] / "cited.scala").unlink()
    doc = make(tmp_path, w)
    checks = by_name(doc)
    assert doc["status"] == "fail"
    assert checks["spec_citations"]["status"] == "unknown"
    assert checks["params_header_equal"]["status"] == "unknown"
    # with no elaboration header and no recorded width, nothing on the RTL side states acc_scale_t
    assert checks["param.acc_scale_t"]["status"] == "unknown"
    assert checks["param.DIM"]["status"] == "pass"  # the RTL facts still state DIM


def test_a_changed_citation_fails(tmp_path, monkeypatch):
    doc = make(tmp_path, world(tmp_path, monkeypatch, cite_line="def x(args: Any*): Int"))
    assert by_name(doc)["spec_citations"]["status"] == "fail"


def test_an_uncalibrated_reader_cannot_derive_the_fpga_memory(tmp_path, monkeypatch):
    doc = make(tmp_path, world(tmp_path, monkeypatch, elab_dts_bytes=0x20000000))
    checks = by_name(doc)
    assert checks["fpga.reader_calibration"]["status"] == "fail"
    assert checks["fpga.target_memory"]["status"] == "unknown"
    assert checks["dram.fpga_vs_board"]["status"] == "unknown"


def test_the_fpga_names_must_be_the_pinned_bitstreams(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch)
    spec = yaml.safe_load(w["spec_path"].read_text())
    spec["fpga"]["platform_config"] = "OtherPlatform"
    w["spec_path"].write_text(yaml.safe_dump(spec))
    assert by_name(make(tmp_path, w))["fpga.identity"]["status"] == "fail"


def test_no_engine_leaves_the_isa_unknown(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch)
    doc = BD.receipt(
        target=TARGET,
        contract_path=w["contract_path"],
        facts_path=w["facts_path"],
        spec_path=w["spec_path"],
        board=None,
        host_package=None,
        dram_base=BASE,
        dram_bytes=16 * GIB,
        work=tmp_path / "work",
    )
    assert doc["status"] == "fail"
    assert by_name(doc)["isa"]["status"] == "unknown"


# ---------------------------------------------------------------------------------------------- readers


def test_header_reader_is_structural():
    header = S.read_header(HEADER.format(dim=16))
    assert header.integer("DIM") == 16 and "FAKE" not in header.defines
    assert header.defines["LONG_MACRO"] == "(x) ((x) + 1)"
    assert header.c_type("elem_t") == {"kind": "int", "signed": True, "bits": 8, "c_type": "int8_t"}
    assert header.c_type("acc_scale_t_bits")["signed"] is False
    assert header.integer("LONG_MACRO") is None


def test_dts_reader_reads_memory_and_isa():
    root = S.read_dts(DTS.format(elab_bytes="0x10000000"))
    assert S.dts_isas(root) == ["rv64imafdc_zicsr_xvendor"]
    memory = S.dts_memory(root)
    assert [(m["base"], m["bytes"], m["enabled"]) for m in memory] == [
        (0x8000000, 0x10000, False),
        (BASE, 0x10000000, True),
    ]
    with pytest.raises(S.SourceReadError):
        S.read_dts("/dts-v1/;\n/ { memory { reg = <0x1")


def test_config_reader_follows_priority_and_modification(tmp_path):
    (tmp_path / "Mem.scala").write_text(MEM_SCALA)
    (tmp_path / "Soc.scala").write_text(SOC_SCALA.format(gib=4))
    reader = S.ConfigSources([tmp_path], config_class="Config", interpolators={"x": 16})
    assert reader.resolve("ElabConfig", "ExtMem", "size").value == 0x10000000
    assert reader.resolve("FpgaConfig", "ExtMem", "size").value == 4 * GIB
    assert reader.resolve("FpgaConfig", "ExtMem", "base").value == BASE  # modified, not replaced
    assert reader.resolve("FpgaConfig", "Unrelated", None).value == 3
    with pytest.raises(S.SourceReadError):
        reader.resolve("Broken", "ExtMem", "size")  # replaced without a size: no value to report
    with pytest.raises(S.SourceReadError):
        reader.resolve("ElabConfig", "NoSuchKey", None)


def test_spike_memory_spans():
    assert BD.spike_memory(["spike", "-m0x80000000:0x1000,0x100000000:0x2000", "x.elf"]) == [
        (BASE, 0x1000),
        (0x100000000, 0x2000),
    ]
    with pytest.raises(BD.BindingError):
        BD.spike_memory(["spike", "-m2048"])


def test_elf_segments(tmp_path):
    assert S.elf_load_segments(_elf(tmp_path / "a.elf", [(BASE, 16), (BASE + 64, 0)])) == [(BASE, 16)]


# ------------------------------------------------------------------------------------- command wiring


def test_binding_command_surface_parses():
    args = cli.build_parser().parse_args(
        [
            "binding",
            "--contract",
            "c.yaml",
            "--rtl-facts",
            "f.json",
            "--board",
            "b",
            "--host-package",
            "h",
            "--image",
            "a.elf",
            "--image",
            "b.elf",
            "--out",
            "r.json",
        ]
    )
    assert args.board == "b" and [p.name for p in args.image] == ["a.elf", "b.elf"]


def test_binding_command_reports_every_failed_check(tmp_path, monkeypatch):
    from merlin.model_check import commands

    w = world(tmp_path, monkeypatch, facts_rows=1024)
    doc = make(tmp_path, w)
    monkeypatch.setattr(BD, "receipt_for", lambda args: doc)
    out = tmp_path / "receipt.json"
    code = cli.main(["binding", "--out", str(out)])
    report = json.loads(out.read_text())
    assert code == R.EXIT_GATE_FAILED and report["status"] == "fail"
    assert report["binding"]["receipt_sha256"] == BD.digest(doc)
    assert {f["check"] for f in report["failures"]} == {c["name"] for c in doc["checks"] if c["status"] != "pass"}
    assert commands  # registered


def test_bundle_build_records_the_receipt_digest(tmp_path, monkeypatch):
    from merlin.model_check import commands

    receipt = {"schema": BD.RECEIPT_SCHEMA, "status": "fail", "checks": [], "mismatches": [], "unknown": []}
    seen = {}
    monkeypatch.setattr(BD, "receipt_for", lambda args: receipt)

    def fake_build(bundle, model, **kwargs):
        seen.update(kwargs)
        return {"programs": {}}

    monkeypatch.setattr(B, "build", fake_build)
    from merlin.common.paths import repo_root

    models = repo_root() / "examples" / "gemmini" / "numerics" / "models.yaml"
    args = cli.build_parser().parse_args(
        [
            "bundle-build",
            "--captures",
            str(tmp_path),
            "--model",
            "resnet50",
            "--host-package",
            "h",
            "--contract",
            "c",
            "--rtl-facts",
            "f",
            "--bundle",
            str(tmp_path / "bundle"),
            "--models",
            str(models),
            "--out",
            "o.json",
        ]
    )
    report = commands.bundle_build(args, R.new_report("bundle-build", model="resnet50", split=None, submission={}))
    written = tmp_path / "bundle" / "binding.json"
    assert hashlib.sha256(written.read_bytes()).hexdigest() == BD.digest(receipt)
    assert seen["binding"] == {"receipt": "binding.json", "receipt_sha256": BD.digest(receipt), "status": "fail"}
    assert report["binding"] == {"receipt_sha256": BD.digest(receipt), "status": "fail"}


def test_bundle_manifest_carries_the_binding(tmp_path):
    bundle = B.Bundle(tmp_path / "bundle")
    bundle.root.mkdir()
    model = Model(name="m", workload="w", chain={"kind": "single"}, programs=(), judged={})
    record = {"receipt": "binding.json", "receipt_sha256": "0" * 64, "status": "pass"}
    B.build(
        bundle, model, target=TARGET, captures=tmp_path, engine={"contract": "c"}, binding=record, log=lambda _: None
    )
    assert bundle.manifest()["binding"] == record
