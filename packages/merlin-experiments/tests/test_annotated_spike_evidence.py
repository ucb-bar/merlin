"""Evidence-only annotated-Spike pass: counter-window derivation and functional identity, with fake tools."""

from __future__ import annotations

import base64
import hashlib
import json
import stat
import sys
import tarfile
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase1 import annotated_spike_evidence as E
from merlin_experiments.phase1 import capsule_timing_archive as archive

LISTING = """
program.elf:     file format elf64-littleriscv

Disassembly of section .text:

0000000080000000 <_start>:
    80000000:\tcsrr\tt0,mhartid
    80000004:\tj\t80000040 <main>

0000000080000040 <main>:
    80000040:\taddi\tsp,sp,-48
    80000044:\tcsrr\ts0,mcycle
    80000048:\tauipc\tra,0x0
    8000004c:\tjalr\t64(ra) # 80000088 <kernel_entry>
    80000050:\tauipc\tra,0x0
    80000054:\tjalr\t60(ra) # 8000008c <kernel_complete>
    80000058:\tcsrr\ta4,mcycle
    8000005c:\tjal\t80000090 <kernel_entry+0x8>
    80000060:\tret

0000000080000088 <kernel_entry>:
    80000088:\tret
"""


def _window(text=LISTING, **kwargs):
    rows = E.parse_disassembly(text)
    options = {"counter_asm": "csrr %0, mcycle", "entry_symbol": "kernel_entry", "completion_symbol": "kernel_complete"}
    return E.counter_window(rows, **{**options, **kwargs})


def test_counter_window_is_the_read_pair_bracketing_entry_and_completion():
    window = _window()
    assert (window["caller"], window["begin_pc"], window["end_pc"]) == ("main", 0x80000044, 0x80000058)
    assert window["occurrence"] == 0 and window["program_counter_reads"] == 2
    assert window["counter_reads"][0]["operands"] == ["s0", "mcycle"]


def test_counter_window_accepts_either_disassembler_spacing():
    spaced = LISTING.replace("\tcsrr\ts0,mcycle", "\tcsrr\ts0, mcycle").replace(
        "\tcsrr\ta4,mcycle", "\tcsrr a4, mcycle"
    )
    assert _window(spaced)["end_pc"] == 0x80000058


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda t: t.replace("    80000050:", "    8000004e:\tcsrr\ta5,mcycle\n    80000050:"), "exactly once"),
        (lambda t: t.replace("\tcsrr\ta4,mcycle", "\taddi\ta4,a4,0"), "exactly once"),
        (lambda t: t.replace("<kernel_complete>", "<kernel_complete+0x4>"), "one function"),
        (
            lambda t: (
                t + "\n0000000080000100 <helper>:\n    80000100:\tjal\t1 <kernel_entry>\n"
                "    80000104:\tjal\t2 <kernel_complete>\n"
            ),
            "one function",
        ),
    ],
)
def test_counter_window_refuses_anything_but_one_bracketing_pair(change, match):
    with pytest.raises(E.EvidenceError, match=match):
        _window(change(LISTING))


def test_counter_template_must_name_its_output_operand():
    with pytest.raises(E.EvidenceError, match="%0"):
        _window(counter_asm="csrr a0, mcycle")


def _tool(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


FUNCTIONAL = """import sys
sys.stdout.write("METRIC cycles 7\\nDONE\\n")
"""
ANNOTATED = """import hashlib, json, sys
args = sys.argv[1:]
option = lambda name: next(a.split("=", 1)[1] for a in args if a.startswith(name + "="))
elf = args[-1]
model = json.load(open(option("--timing-model")))
report = {
    "schema": "merlin.annotated_spike_report.v1",
    "elf_sha256": hashlib.sha256(open(elf, "rb").read()).hexdigest(),
    "domain": model["domain"],
    "functional_status": "PASS",
    "timing_complete": False,
    "feature_counts": {"target.commands": 3, "target.class.load": 2},
    "unresolved": ["target:price:load.latency_cycles"],
    "counter_windows": [{"instructions": 7}],
    "scope_binding": json.load(open(option("--timing-scope-binding"))),
    "argv": args,
}
json.dump(report, open(option("--timing-report"), "w"))
sys.stdout.write("METRIC cycles 7\\nDONE\\n" if "DIVERGE" not in elf else "changed\\n")
"""


def _selection():
    record = {
        "schema": "merlin.kernel_call_timing.v1",
        "selection_sha256": "a" * 64,
        "boundary": "immediately before kernel entry through explicitly selected completion",
        "counter_asm": "csrr %0, mcycle",
        "max_delta_cycles": 1000,
        "counter_semantics": {
            "unit": "cycles",
            "width_bits": 64,
            "wrap": "modulo",
            "clock": "selected-clock",
            "source_pins": [{"path": "/original/source", "sha256": "b" * 64}],
        },
    }
    return record, base64.b64encode(json.dumps(record).encode()).decode()


def _run(tmp_path: Path, name: str = "A", *, functional_only: str | None = None) -> tuple[Path, dict]:
    """One run capsule; ``functional_only`` instead leaves only the functional tier, with that console."""
    tools = tmp_path / "tools"
    tools.mkdir(exist_ok=True)
    spike = _tool(tools / "spike", FUNCTIONAL)
    annotated = _tool(tools / "annotated-spike", ANNOTATED)
    objdump = _tool(tools / "objdump", f"import sys\nsys.stdout.write({LISTING!r})\n")
    for file, text in {"lib.so": "functional extension", "provider.so": "provider", "bundle.json": "{}"}.items():
        (tools / file).write_text(text)
    (tools / "source.json").write_text(json.dumps({"sha256": "c" * 64}))
    (tools / "device.json").write_text(json.dumps({"target": {"hardware": {"dim": 4}}}))
    (tools / "contract.yaml").write_text(
        yaml.safe_dump(
            {
                "harness_abi": {"entry_symbol": "kernel_entry", "fence_symbol": "kernel_complete"},
                "logical_harness": {"cycle_counter_asm": "csrr %0, mcycle"},
            }
        )
    )
    run = tmp_path / "run"
    capsule = run / "_qa_work" / "runs_a" / "runs" / "suite" / name
    generated = capsule / "generated"
    (capsule / "artifacts").mkdir(parents=True)
    generated.mkdir(parents=True)
    selection, encoded = _selection()
    (generated / "harness.c").write_text(
        f"/* merlin.kernel_call_timing.v1 selection_sha256={'a' * 64} "
        f"boundary=entry-through-completion record_b64={encoded} */\n"
    )
    (generated / "package_kernel.elf").write_bytes(f"gsim program {name}".encode())
    for file in ("input.interface.mlir", "command_buffer.json", "instruction_trace.json", "numeric_report.yaml"):
        (generated / file).write_text("{}\n")
    if functional_only is None:
        (capsule / "artifacts" / "rtl_console.log").write_text("METRIC cycles 90\nDONE\n")
    else:
        (capsule / "artifacts" / "functional_console.log").write_text(functional_only)
    elf = generated / "package_kernel.elf"
    spike_sha = hashlib.sha256(spike.read_bytes()).hexdigest()
    elf_sha = hashlib.sha256(elf.read_bytes()).hexdigest()
    for folder, executable, argv in (
        ("l2", spike_sha, [str(spike), f"--extlib={tools / 'lib.so'}", "--extension=x", "--isa=rv64imafdc"]),
        ("l3", "e" * 64, ["/gsim"]),
    ):
        if functional_only is not None and folder == "l3":
            continue
        record = generated / "invocations" / folder
        record.mkdir(parents=True)
        (record / "invocation.json").write_text(
            json.dumps(
                {
                    "schema": "merlin.invocation_record.v1",
                    "argv": [*argv, str(elf)],
                    "cwd": str(tmp_path),
                    "executable": {"path": argv[0], "sha256": executable},
                    "inputs": [{"path": str(elf), "sha256": elf_sha}],
                    "outputs": [],
                    "finished_ns": 1 if folder == "l2" else 2,
                }
            )
        )
    (capsule / "capsule_result.json").write_text(
        json.dumps(
            {
                "capsule": name,
                "status": "pass",
                "command_buffer_artifact": {"artifact_sha256": "d" * 64, "command_buffer": {"tensors": {}}},
                "original_compiler_child": {"candidate_sha256": "f" * 64},
                "tiers": {
                    "L2": {"status": "pass", "engine": "spike", "sim_provenance": {"sha256": spike_sha}},
                    **(
                        {}
                        if functional_only is not None
                        else {
                            "L3": {
                                "status": "pass",
                                "engine": "gsim",
                                "derived_from_rtl": True,
                                "cycle_accurate": True,
                                "cycles": 90,
                                "sim_provenance": {"sha256": "e" * 64},
                            }
                        }
                    ),
                },
            }
        )
    )
    config = {
        "schema": E.CONFIG_SCHEMA,
        "annotated_spike": str(annotated),
        "annotated_source_manifest": str(tools / "source.json"),
        "timing_provider": str(tools / "provider.so"),
        "provider_bundle_manifest": str(tools / "bundle.json"),
        "device_profile": str(tools / "device.json"),
        "objdump": str(objdump),
        "contract": str(tools / "contract.yaml"),
    }
    (tools / "config.json").write_text(json.dumps(config))
    return run, {"config": tools / "config.json", "capsule": capsule, "tools": tools, "selection": selection}


def test_pass_observes_the_gsim_program_with_the_runs_own_functional_configuration(tmp_path):
    run, fixture = _run(tmp_path)
    assert E.main([str(run), "--config", str(fixture["config"])]) == 0
    summary = json.loads((run / "annotated_spike_evidence.json").read_text())
    assert summary["counts"] == {
        "programs": 1,
        "errors": 0,
        "functional_identical": 1,
        "retained_consoles_checked": 0,
        "retained_consoles_reproduced": 0,
        "reports": 1,
    }
    (observation,) = summary["observations"]
    assert (observation["begin_pc"], observation["end_pc"], observation["window_instructions"]) == (
        0x80000044,
        0x80000058,
        7,
    )
    folder = fixture["capsule"] / E.EVIDENCE_DIR / E.PROGRAM_ROLE
    report = json.loads((folder / "report.json").read_text())
    tools = fixture["tools"]
    # The functional extension loads first and the provider immediately after; the program comes last.
    assert report["argv"][:4] == [
        f"--extlib={tools / 'lib.so'}",
        f"--extlib={tools / 'provider.so'}",
        "--extension=x",
        "--isa=rv64imafdc",
    ]
    assert report["argv"][-1].endswith("package_kernel.elf")
    binding = json.loads((folder / "scope_binding.json").read_text())
    review = json.loads((folder / "counter_window_review.json").read_text())
    assert binding["evidence_sha256"] == E.canonical_digest(review)
    assert binding["scope_sha256"] == E.kernel_call_scope(fixture["selection"])
    model = json.loads((run / "count_only_model.json").read_text())
    assert model["parameters"] == {} and model["target"]["prices"] == {}
    assert model["libraries"] == [
        hashlib.sha256((tools / name).read_bytes()).hexdigest() for name in ("lib.so", "provider.so")
    ]
    # The audit indexes the observation beside its exact program, so it enters the training subset.
    row = archive.audit(run)["records"][0]
    assert row["training_eligible"] is True
    assert row["artifacts"]["annotated_spike_report"]["path"].endswith("gsim_program/report.json")
    with tarfile.open(archive.training_archive(run, tmp_path / "training.tar")["archive"]) as tar:
        assert "run/" + row["artifacts"]["annotated_spike_report"]["path"] in tar.getnames()


def test_a_console_that_differs_under_observation_fails_the_pass(tmp_path):
    run, fixture = _run(tmp_path, name="DIVERGE")
    assert E.main([str(run), "--config", str(fixture["config"]), "--out", str(tmp_path / "evidence")]) == 1
    summary = json.loads((tmp_path / "evidence" / "annotated_spike_evidence.json").read_text())
    assert summary["observations"][0]["functional_identical"] is False
    # Written elsewhere, the observation is not indexed beside the program.
    assert archive.audit(run)["records"][0]["artifacts"]["annotated_spike_report"] is None


def test_a_stale_report_for_another_program_is_not_indexed(tmp_path):
    run, fixture = _run(tmp_path)
    assert E.main([str(run), "--config", str(fixture["config"])]) == 0
    (fixture["capsule"] / "generated" / "package_kernel.elf").write_bytes(b"rebuilt")
    assert archive.audit(run)["records"][0]["artifacts"]["annotated_spike_report"] is None


@pytest.mark.parametrize(("retained", "reproduced"), [("METRIC cycles 7\nDONE\n", True), ("METRIC cycles 8\n", False)])
def test_the_functional_tiers_own_program_must_reproduce_its_retained_console(tmp_path, retained, reproduced):
    run, fixture = _run(tmp_path)
    _run(tmp_path, "SCREENED", functional_only=retained)
    summary = E.collect(run, fixture["config"], tmp_path / "evidence")
    rows = {row["role"]: row for row in summary["observations"]}
    assert set(rows) == {E.PROGRAM_ROLE, E.FUNCTIONAL_PROGRAM_ROLE}
    assert rows[E.FUNCTIONAL_PROGRAM_ROLE]["capsule"] == "SCREENED"
    assert rows[E.FUNCTIONAL_PROGRAM_ROLE]["retained_console_reproduced"] is reproduced
    assert summary["counts"]["retained_consoles_checked"] == 1
