"""``merlin experiment inspect <candidate> --group gN``: one group rebuilt, its IR per stage, its program, and
an instruction trace -- or "not available for this target" when the target provides no hook.

The fixture is a tiny one: a measured job directory whose target's group-program build is replaced by
one that writes the group's statement products and lowers a one-op module through Merlin's LLVM route
IN A CHILD PROCESS, as a package's own compiler would. So the trace the inspection reads is a real one,
reached across a process boundary, and the functional model is a stand-in that writes the execution log
its real counterpart's options ask for.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from merlin_experiments import cli
from merlin_experiments import group_inspect as GI

from merlin.common.paths import python_import_roots

_MODULE = """
func.func @forward(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %e = tensor.empty() : tensor<4xf32>
  %r = linalg.add ins(%a, %b : tensor<4xf32>, tensor<4xf32>) outs(%e : tensor<4xf32>) -> tensor<4xf32>
  return %r : tensor<4xf32>
}
"""

_FAKE_FUNCTIONAL_MODEL = """
import sys
if "-g" in sys.argv:  # the PC histogram run: the simulator's own banner, then address/count lines on stderr
    sys.stderr.write("PC Histogram size: 2\\n0x10 5\\n0x14 7\\n")
    sys.exit(0)
log = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--log="))
count = int(next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--instructions=")))
with open(log, "w") as fh:
    for i in range(count):
        fh.write(f"core   0: 0x{0x80000000 + 4 * i:016x} (0x00000013) addi zero, zero, 0\\n")
print("ran", sys.argv[-1])
"""


def _job(tmp_path: Path, *, functional: bool = True) -> Path:
    job = tmp_path / "job"
    (job / "package").mkdir(parents=True)
    model = tmp_path / "functional_model"  # an executable, as the real simulator is: the log options follow it
    model.write_text(f"#!{sys.executable}\n" + _FAKE_FUNCTIONAL_MODEL)
    model.chmod(0o755)
    machine = {"kind": "paired", "local": {"kind": "spike", "command": [str(model), "--isa=fixture"]}, "timing": {}}
    document = {
        "target": "fixture",
        "package_sha256": "p",
        "build_options": {"model_capsule": str(tmp_path / "capsule"), "machine": "mach", "header": "h.h"},
        "machine": machine if functional else {"kind": "gsim"},
    }
    (job / "job.json").write_text(json.dumps(document))
    return job


@pytest.fixture
def group_build(monkeypatch):
    """The target hooks: a whole-model driver exists, and a group program is built as described above."""
    from merlin.perf import whole_model_group_timing as GT

    monkeypatch.setattr("merlin.runtime.backends.base.whole_model_driver", lambda target: object())
    asked = {}

    def build_group_programs(package_dir, groups, *, out, ask_only, keep_statement, **kwargs):
        asked.update(groups=list(groups), ask_only=ask_only, keep_statement=keep_statement, **kwargs)
        (group,) = groups
        out = Path(out)
        for rel, text in {
            f"lower/g{group}.iface.mlir": "// the interface capsule the package was given",
            f"lower/g{group}.artifact.txt": "llvm.func @kernel()",
            f"lower/g{group}.generated/command_buffer.json": '{"commands": []}',
            f"lower/g{group - 1}.iface.mlir": "// another group's: never shown",
            f"objects/g{group}.o": "\x7fELF",
            f"g{group}/program/program.c": "int main(void) { return 0; }",
            f"g{group}/program/program.elf": "\x7fELF",
        }.items():
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            (out / rel).write_text(text)
        script = f"from merlin.llvmlower.lower import lower_model\nlower_model({_MODULE!r}, {str(out / 'pkg')!r}, targets=())\n"
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(p) for p in python_import_roots()))
        done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=600)
        assert done.returncode == 0, done.stderr
        here = out / f"g{group}"
        return {
            group: {
                "group": group,
                "on": "package",
                "interface": str(out / f"lower/g{group}.iface.mlir"),
                "command_buffer": str(out / f"lower/g{group}.generated/command_buffer.json"),
                "program_source": str(here / "program/program.c"),
                "elf": str(here / "program/program.elf"),
            }
        }

    monkeypatch.setattr(GT, "build_group_programs", build_group_programs)
    return asked


def test_one_group_is_rebuilt_alone_and_its_ir_program_and_trace_are_shown(group_build, tmp_path, capsys):
    work = tmp_path / "work"
    argv = [
        "inspect",
        str(_job(tmp_path)),
        "--group",
        "g2",
        "--stage",
        "mlir:one-shot-bufferize",
        "--trace",
        "--run-to",
        "7",
        "--out",
        str(work),
        "--json",
    ]
    assert cli.main(argv) == 0
    result = json.loads(capsys.readouterr().out)
    assert group_build["groups"] == [2] and group_build["ask_only"] and group_build["keep_statement"]
    assert result["answered_by"] == "package" and result["work"] == str(work)
    products = set(result["products"])
    assert {
        "lower/g2.iface.mlir",
        "lower/g2.artifact.txt",
        "lower/g2.generated/command_buffer.json",
        "objects/g2.o",
    } <= products
    assert not any(p.startswith("lower/g1") for p in products)  # another group's file is not this group's
    stage = result["stage"]["files"]
    assert stage and "memref" in Path(stage[0]).read_text() and "/mlir/p" in stage[0]  # the child's native pass
    trace = json.loads(Path(result["trace"]).read_text())
    assert trace["outcome"] == "completed" and any(e["stage"] == "llvm-final" for e in trace["stages"])
    log = result["instruction_trace"]
    assert log["available"] and log["log_lines"] == 7 and "--instructions=7" in log["argv"]


def test_a_product_is_shown_by_name_and_the_text_report_prints_its_paths(group_build, tmp_path, capsys):
    argv = ["inspect", str(_job(tmp_path)), "--group", "g2", "--stage", "command_buffer", "--out", str(tmp_path / "w")]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "group g2 of" in out and "trace:" in out and "lower/g2.generated/command_buffer.json" in out
    assert '{"commands": []}' in out


def test_a_candidate_without_a_functional_model_has_no_instruction_trace(group_build, tmp_path, capsys):
    argv = [
        "inspect",
        str(_job(tmp_path, functional=False)),
        "--group",
        "g2",
        "--trace",
        "--out",
        str(tmp_path / "w"),
        "--json",
    ]
    assert cli.main(argv) == 0
    log = json.loads(capsys.readouterr().out)["instruction_trace"]
    assert not log["available"] and "not available for this target" in log["why"]


def test_a_target_without_a_whole_model_driver_is_not_available(monkeypatch, tmp_path, capsys):
    def no_driver(target):
        raise NotImplementedError(f"backend for target {target!r} declares no whole_model_driver")

    monkeypatch.setattr("merlin.runtime.backends.base.whole_model_driver", no_driver)
    assert cli.main(["inspect", str(_job(tmp_path)), "--group", "g2", "--out", str(tmp_path / "w")]) == 2
    assert "not available for this target" in capsys.readouterr().out


def test_without_group_inspect_still_reads_an_experiment_definition(capsys):
    assert cli.main(["inspect", "no-such-experiment"]) == 2  # the existing verb's own refusal
    assert "unknown experiment" in capsys.readouterr().err


def test_a_package_directory_needs_its_build_options_named(tmp_path):
    (tmp_path / "pkg").mkdir()
    with pytest.raises(GI.InspectError, match="--target"):
        GI.resolve_candidate(tmp_path / "pkg")
    options = tmp_path / "options.yaml"
    options.write_text(textwrap.dedent("model_capsule: c\nmachine: m\nheader: h.h\n"))
    import argparse

    resolved = GI.resolve_candidate(tmp_path / "pkg", argparse.Namespace(target="fixture", build_options=options))
    assert (
        resolved["options"] == {"model_capsule": "c", "machine": "m", "header": "h.h"} and resolved["machine"] is None
    )


# ------------------------------------------------------------------------------- source attribution

_FAKE_SYMBOLIZER = """
import json, sys
assert "--output-style=JSON" in sys.argv and any(a.startswith("--obj=") for a in sys.argv)
for line in sys.stdin:
    if line.strip():
        frame = {"FunctionName": "main", "FileName": "program.c", "Line": int(line, 16) // 4, "Column": 1}
        print(json.dumps({"Address": line.strip(), "Symbol": [frame]}))
"""


def _twins(root: Path, *, literal: str = "group program") -> dict[str, Path]:
    """The program and its debug companion, built by a real compiler from one source, the companion with
    the debug option the group build adds; ``literal`` changes the companion's source when it differs."""
    import shutil

    from merlin.perf.whole_model_group_timing import DEBUG_INFO_OPTION

    clang = shutil.which("clang")
    if not clang:
        pytest.skip("clang unavailable")
    toolchain = root / "toolchain"
    toolchain.mkdir(parents=True, exist_ok=True)
    (toolchain / "clang").symlink_to(clang)
    built = {}
    for name, flags, text in (("program", [], "group program"), ("debug", [DEBUG_INFO_OPTION], literal)):
        source = root / f"{name}.c"
        source.write_text(f'extern int puts(const char*);\nint main(void){{return puts("{text}");}}\n')
        obj, elf = root / f"{name}.o", root / f"{name}.elf"
        subprocess.run([clang, "-O2", "-fno-builtin", *flags, "-c", str(source), "-o", str(obj)], check=True)
        subprocess.run([clang, "-Wl,--build-id=none", str(obj), "-o", str(elf)], check=True)
        built[name] = (obj, elf)
    return {
        "toolchain": toolchain,
        "elf": built["program"][1],
        "object": built["program"][0],
        "debug_elf": built["debug"][1],
        "debug_object": built["debug"][0],
    }


@pytest.fixture
def companion_build(group_build, monkeypatch, tmp_path):
    """The group build above, whose record names a real program, its real debug companion and the compiler
    of a toolchain that ships the (stand-in) LLVM symbolizer -- and spies on the companion module."""
    from merlin.perf import debug_companion as DC
    from merlin.perf import whole_model_group_timing as GT

    twins = _twins(tmp_path / "twins")
    symbolizer = twins["toolchain"] / "llvm-symbolizer"
    symbolizer.write_text(f"#!{sys.executable}\n" + _FAKE_SYMBOLIZER)
    symbolizer.chmod(0o755)
    base = GT.build_group_programs

    def with_companion(package_dir, groups, **kwargs):
        records = base(package_dir, groups, **kwargs)
        for record in records.values():
            record.update(elf=str(twins["elf"]), program_object=str(twins["object"]))
            if kwargs.get("debug_companion"):
                record["debug_companion"] = {
                    "elf": str(twins["debug_elf"]),
                    "program_object": str(twins["debug_object"]),
                    "compiler": str(twins["toolchain"] / "clang"),
                    "option": "-g",
                }
        return records

    monkeypatch.setattr(GT, "build_group_programs", with_companion)
    calls = {"verify": [], "attribute": []}
    verify, attribute = DC.verify_debug_companion, DC.attribute_symbolized_pcs

    def spy_verify(control, companion, *, relocatable=False):
        calls["verify"].append(relocatable)
        return verify(control, companion, relocatable=relocatable)

    def spy_attribute(counts, records):
        calls["attribute"].append(dict(counts))
        return attribute(counts, records)

    monkeypatch.setattr(DC, "verify_debug_companion", spy_verify)
    monkeypatch.setattr(DC, "attribute_symbolized_pcs", spy_attribute)
    return {"twins": twins, "calls": calls, "symbolizer": symbolizer}


def test_trace_attributes_the_pc_histogram_through_the_verified_debug_companion(companion_build, tmp_path, capsys):
    argv = ["inspect", str(_job(tmp_path)), "--group", "g2", "--trace", "--out", str(tmp_path / "w"), "--json"]
    assert cli.main(argv) == 0
    result = json.loads(capsys.readouterr().out)
    calls = companion_build["calls"]
    assert sorted(calls["verify"]) == [False, True]  # the image and the object were both compared
    assert calls["attribute"] == [{0x10: 5, 0x14: 7}]  # the functional model's own histogram
    attribution = result["source_attribution"]
    assert attribution["status"] == "attributed" and attribution["total"] == 12
    assert attribution["symbolizer"] == str(companion_build["symbolizer"])
    assert attribution["functions"] == {"main": 12}
    assert [row["executions"] for row in attribution["lines"]] == [7, 5]
    assert json.loads(Path(attribution["file"]).read_text())["total"] == 12


def test_the_group_build_is_asked_for_a_companion_only_when_tracing(group_build, tmp_path, capsys):
    job = _job(tmp_path)
    assert cli.main(["inspect", str(job), "--group", "g2", "--out", str(tmp_path / "a")]) == 0
    assert group_build["debug_companion"] is False
    assert cli.main(["inspect", str(job), "--group", "g2", "--trace", "--out", str(tmp_path / "b")]) == 0
    assert group_build["debug_companion"] is True


def test_a_toolchain_without_a_symbolizer_leaves_the_attribution_unknown(companion_build, tmp_path, capsys):
    companion_build["symbolizer"].unlink()
    argv = ["inspect", str(_job(tmp_path)), "--group", "g2", "--trace", "--out", str(tmp_path / "w"), "--json"]
    assert cli.main(argv) == 0
    attribution = json.loads(capsys.readouterr().out)["source_attribution"]
    assert attribution["status"] == "UNKNOWN" and "llvm-symbolizer" in attribution["why"]
    assert companion_build["calls"]["attribute"] == []  # nothing was attributed without the target's tool


def test_a_companion_that_is_not_the_program_is_refused(group_build, monkeypatch, tmp_path, capsys):
    from merlin.perf import whole_model_group_timing as GT

    twins = _twins(tmp_path / "twins", literal="another program")
    base = GT.build_group_programs

    def with_changed_companion(package_dir, groups, **kwargs):
        records = base(package_dir, groups, **kwargs)
        for record in records.values():
            record.update(elf=str(twins["elf"]), program_object=str(twins["object"]))
            record["debug_companion"] = {
                "elf": str(twins["debug_elf"]),
                "program_object": str(twins["debug_object"]),
                "compiler": str(twins["toolchain"] / "clang"),
            }
        return records

    monkeypatch.setattr(GT, "build_group_programs", with_changed_companion)
    argv = ["inspect", str(_job(tmp_path)), "--group", "g2", "--trace", "--out", str(tmp_path / "w"), "--json"]
    assert cli.main(argv) == 0
    attribution = json.loads(capsys.readouterr().out)["source_attribution"]
    assert attribution["status"] == "UNKNOWN" and "not the program" in attribution["why"]
