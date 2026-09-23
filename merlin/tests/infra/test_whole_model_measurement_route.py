"""The whole-model measurement route, held to the rules the batched route was already held to.

Every FireSim row this repository has on disk came through the whole-model route, and that route
admitted a run on evidence the batched route would have refused:

1. the per-group checksum was an ADDITIVE SUM, which is permutation-blind.  Reorder a group's
   output -- a transpose, a stride, a tiling bug, the failures a tensor compiler actually produces
   -- and the number does not move.  Nor does it move for a pair of compensating +k/-k errors.  The
   tests below run both mutations through the real admission rule and require the sum to accept
   them (it does, which is the finding) and the digest to refuse them;
2. admission walked the groups the POLICY declared and never the set the program PRINTED, so a
   group nobody declared was never looked at;
3. a declared group the window never printed was reported with the sentinel ``-1`` -- a value a
   signed checksum can legitimately take;
4. ``k=v`` tokens were parsed two different ways by the two halves of one parser, one raising on a
   duplicate key and the other silently keeping the first;
5. nothing recorded WHAT KIND OF SPAN a cycle number counted, and a headline ratio divided a single
   contiguous wall window by a sum of 71 per-call deltas;
6. a finished job could not be bound to the bytes that ran it.

Each test below is a mutation of a real mechanism, not of a fixture: the UARTs are rendered by the
harness's own renderer off the same format strings its C ``printf`` calls are built from, and the
verdicts come from :mod:`merlin.perf.firesim_batch` rather than from a copy of its rules.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from merlin.common.paths import merlin_dir
from merlin.perf import firesim_batch as FB
from merlin.perf import firesim_checkpoint as FC
from merlin.perf.firesim_receipt import FireSimReceiptError, _key_values

_POLICIES = merlin_dir() / "contract" / "firesim_uart_policies"
#: The policy the whole-model cycle claim was admitted against before this work: its checksums are
#: additive sums and its ``value_key`` is ``sum``.  Kept as the description of the retired program.
_RETIRED_POLICY = _POLICIES / "resnet50_group_model_cycle_claim_v2.json"
#: The policy for the program as it stands: order-sensitive digests, and a declared window kind.
_POLICY = _POLICIES / "resnet50_group_model_cycle_claim_fnv1a_v2.json"


def _harness():
    """The whole-model harness, loaded from its path: it is a script, not a package module."""
    path = merlin_dir() / "experiments" / "gemmini_perf_bench" / "scripts" / "group_model_program.py"
    spec = importlib.util.spec_from_file_location("group_model_program", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


# --------------------------------------------------------------- a tiny whole-model program
def _model() -> dict:
    """Four device groups over named buffers: the shape :func:`emulate` and :func:`render` take."""
    rng = np.random.default_rng(7)
    arrays = {
        "IMAGE_DATA": rng.integers(-128, 128, size=(1, 6, 6, 2), dtype=np.int8),
        "W_g1": rng.integers(-8, 8, size=(3, 3, 2, 4), dtype=np.int8),
        "BIAS_g1": rng.integers(-50, 50, size=4, dtype=np.int32),
        "W_g4": rng.integers(-8, 8, size=(4, 5), dtype=np.int8),
        "GOLDEN": np.arange(5, dtype=np.float32),
    }
    steps = [
        {"kind": "conv2d", "group": 1, "in": "IMAGE", "out": "B_g1", "weight": "W_g1", "bias": "BIAS_g1", "relu": True,
         "scale": 0.05, "in_dim": 6, "ci": 2, "n": 4, "out_dim": 6, "stride": 1, "padding": 1, "kernel": 3,
         "pool": {"size": 0, "stride": 0, "padding": 0}},
        {"kind": "sum", "group": 2, "lhs": "B_g1", "rhs": "B_g1", "out": "B_g2", "rows": 36, "cols": 4,
         "lhs_load": 0.37, "rhs_load": 0.61, "readout": 1.0, "relu": False, "bound_lsb": 1},
        {"kind": "mean", "group": 3, "in": "B_g2", "out": "B_g3", "rows": 4, "window": 36, "multiplier": 0.04},
        {"kind": "matmul", "group": 4, "in": "B_g3", "out": "B_g4", "weight": "W_g4", "bias": None, "relu": False,
         "scale": None, "dequantize": 0.5, "m": 1, "k": 4, "n": 5},
    ]  # fmt: skip
    buffers = [
        {"name": "B_g1", "elements": 144, "ctype": "elem_t"},
        {"name": "B_g2", "elements": 144, "ctype": "elem_t"},
        {"name": "B_g3", "elements": 4, "ctype": "elem_t"},
        {"name": "B_g4", "elements": 5, "ctype": "acc_t"},
    ]
    return {"steps": steps, "buffers": buffers, "arrays": arrays, "classes": 5, "groups": 5, "device_groups": 4}


def _oracle() -> tuple[object, dict, dict, dict]:
    harness = _harness()
    model = _model()
    emulation = harness.emulate(model)
    return harness, model, emulation, harness.group_checksums(model, emulation)


def _policy_for(rows: dict, *, value_key: str, window_kind: str | None = FB.WINDOW_KIND_SUM_OF_CALLS):
    """A v2 policy declaring one window, checked against ``value_key`` of the harness's own oracle."""
    return FB.BatchValidationPolicy(
        policy_id="synthetic_whole_model",
        workload="synthetic-workload",
        checksum_line=FB.ChecksumLine(prefix="GM_GROUP", value_key=value_key, group_token_offset=1),
        per_window=(
            FB.WindowPolicy(
                label="group_model",
                markers=(),
                checksums=tuple(sorted((group, values[value_key]) for group, values in rows.items())),
                window_kind=window_kind,
            ),
        ),
    )


def _uart(harness, model, emulation, rows: dict) -> str:
    return (
        "\n".join(
            harness.uart_lines(
                harness.program_steps(model),
                cycles=987654,
                checksums={group: values["sum"] for group, values in rows.items()},
                digests={group: values["fnv1a"] for group, values in rows.items()},
                argmax=emulation["argmax"],
                want=emulation["want"],
                cosine_ppm=int(emulation["cosine"] * 1000000.0),
            )
        )
        + "\n"
    )


def _verdict(uart: str, policy) -> FB.WindowAdmission:
    return FB.admit_window(uart, policy.window("group_model"), policy.checksum_line)


# --------------------------------------------------------------- MUTATION 1: a permuted output
def test_a_permuted_group_output_clears_the_additive_sum_and_is_refused_by_the_digest() -> None:
    """THE mutation the whole-model route could not see, run through the real admission rule.

    Group 1's output is REORDERED and nothing else is touched: no value is added, removed or
    changed.  Under the additive sum every group still matches and the window passes -- which is the
    finding, not an artefact of the fixture -- and under the order-sensitive digest the same window
    is ``fail`` and names group 1.
    """
    harness, model, emulation, oracle = _oracle()
    permuted = dict(emulation["values"])
    permuted["B_g1"] = np.roll(permuted["B_g1"], 1)
    assert not np.array_equal(permuted["B_g1"], emulation["values"]["B_g1"]), "the output really moved"
    ran = harness.group_checksums(model, {"values": permuted})

    assert ran["1"]["sum"] == oracle["1"]["sum"], "an additive sum is blind to a permutation"
    assert ran["1"]["fnv1a"] != oracle["1"]["fnv1a"], "the digest is not"

    uart = _uart(harness, model, emulation, ran)
    by_sum = _verdict(uart, _policy_for(oracle, value_key="sum"))
    by_digest = _verdict(uart, _policy_for(oracle, value_key="fnv1a"))

    assert by_sum.status == FB.PASS, "the retired rule admits a reordered output"
    assert by_digest.status == FB.FAIL
    assert [group for group, _observed, _expected in by_digest.checksum_mismatches] == ["1"]


def test_two_compensating_errors_clear_the_additive_sum_and_are_refused_by_the_digest() -> None:
    """The other shape a sum cannot see: +k in one element and -k in another."""
    harness, model, emulation, oracle = _oracle()
    damaged = dict(emulation["values"])
    values = np.array(damaged["B_g2"])
    values[0] += 7
    values[1] -= 7
    damaged["B_g2"] = values
    ran = harness.group_checksums(model, {"values": damaged})

    assert ran["2"]["sum"] == oracle["2"]["sum"]
    assert ran["2"]["fnv1a"] != oracle["2"]["fnv1a"]

    uart = _uart(harness, model, emulation, ran)
    assert _verdict(uart, _policy_for(oracle, value_key="sum")).status == FB.PASS
    assert _verdict(uart, _policy_for(oracle, value_key="fnv1a")).status == FB.FAIL


def test_an_unmutated_run_still_passes_under_the_digest() -> None:
    """The control: nothing else in these UARTs is what rejects the mutated ones."""
    harness, model, emulation, oracle = _oracle()
    uart = _uart(harness, model, emulation, oracle)

    verdict = _verdict(uart, _policy_for(oracle, value_key="fnv1a"))

    assert verdict.status == FB.PASS and verdict.cycles == 987654


def test_the_device_c_and_the_host_oracle_compute_the_same_digest(tmp_path: Path) -> None:
    """The two ends of the check, compiled and run rather than asserted to agree.

    ``group_digest`` is the host oracle a policy is generated from; ``checksum_elem`` /
    ``checksum_acc`` are what the emitted program runs on the device.  A digest whose two
    implementations disagree is worse than no digest: every run would read as wrong.  The helpers
    are lifted out of the program this harness really renders, not retyped here.
    """
    harness = _harness()
    source = harness.render(_model())
    start = source.index("static long long sum_elem")
    end = source.index("/* Our own schedules")
    helpers = source[start:end]
    rng = np.random.default_rng(11)
    elems = rng.integers(-128, 128, size=257, dtype=np.int64)
    accs = rng.integers(-(2**30), 2**30, size=129, dtype=np.int64)

    program = tmp_path / "digest.c"
    program.write_text(
        "#include <stdint.h>\n#include <stdio.h>\n#include <stddef.h>\n"
        "typedef signed char elem_t;\ntypedef int32_t acc_t;\n"
        + helpers
        + "static const elem_t E[] = {"
        + ",".join(str(int(v)) for v in elems)
        + "};\n"
        + "static const acc_t A[] = {"
        + ",".join(str(int(v)) for v in accs)
        + "};\n"
        + "int main(void){\n"
        f'  printf("%lld %lld\\n", checksum_elem(E, {elems.size}), checksum_acc(A, {accs.size}));\n'
        "  return 0;\n}\n",
        encoding="utf-8",
    )
    binary = tmp_path / "digest"
    built = subprocess.run(["cc", "-O2", "-std=gnu99", str(program), "-o", str(binary)], capture_output=True, text=True)
    if built.returncode != 0:  # pragma: no cover - a host without a C compiler
        pytest.skip(f"no working host C compiler: {built.stderr[-200:]}")
    printed = subprocess.run([str(binary)], capture_output=True, text=True, check=True).stdout.split()

    assert [int(token) for token in printed] == [harness.group_digest(elems), harness.group_digest(accs)]


def test_the_digest_does_not_depend_on_how_wide_the_element_type_is() -> None:
    """Why the device widens each element to 64 bits before mixing it.

    The same values digest to the same number whether they arrive as an int8 output or as an
    accumulator, so one host function checks both and neither side holds a width as a constant --
    which is what keeps this derivable rather than a fact about one target's ``gemmini_params.h``.
    """
    harness = _harness()
    values = np.array([-128, -1, 0, 1, 127], dtype=np.int64)

    assert harness.group_digest(values.astype(np.int8)) == harness.group_digest(values.astype(np.int32))
    assert harness.group_digest(values) == harness.group_digest(values.astype(np.int8))


# --------------------------------------------------------------- the tracked policies
def test_the_tracked_whole_model_policy_declares_the_order_sensitive_key_and_its_window_kind() -> None:
    document = json.loads(_POLICY.read_text(encoding="utf-8"))
    window = document["per_window"][0]

    assert document["checksum_line"]["value_key"] == "fnv1a", "a cycle claim is not admitted on a sum"
    assert window["window_kind"] == FB.WINDOW_KIND_SUM_OF_CALLS, "this program sums per-call deltas"
    assert sorted(int(group) for group in window["checksums"]) == list(range(1, 72)), "every device group"
    assert window["markers"] == ["GM_ARGMAX got=21 want=21 agrees=1", "GM_COSINE_PPM 997981"]


def test_the_retired_policy_can_no_longer_admit_a_run_of_the_current_program() -> None:
    """A FINDING, asserted so it cannot be forgotten rather than left in a report.

    The retired policy reads ``sum=``.  The current program still prints that token, so the retired
    document is not silently broken -- but it checks the permutation-blind number, which is why it
    is retired and why it must not be handed to a new cycle claim.  What a reader needs from this
    test is the pair: the old document is still parseable, and it is still blind.
    """
    harness, model, emulation, oracle = _oracle()
    retired = json.loads(_RETIRED_POLICY.read_text(encoding="utf-8"))
    assert retired["checksum_line"]["value_key"] == "sum"
    assert "window_kind" not in retired["per_window"][0], "it predates the field; it seals as UNKNOWN"

    permuted = dict(emulation["values"])
    permuted["B_g1"] = np.roll(permuted["B_g1"], 1)
    ran = harness.group_checksums(model, {"values": permuted})

    assert _verdict(_uart(harness, model, emulation, ran), _policy_for(oracle, value_key="sum")).status == FB.PASS


def test_a_policy_declaring_the_digest_refuses_a_run_that_published_no_digest() -> None:
    """Fail-closed on the version skew, rather than reading an absent check as a passing one."""
    harness, model, emulation, oracle = _oracle()
    old_style = "\n".join(
        harness.uart_lines(
            harness.program_steps(model),
            cycles=987654,
            checksums={group: values["sum"] for group, values in oracle.items()},
            argmax=emulation["argmax"],
            want=emulation["want"],
            cosine_ppm=int(emulation["cosine"] * 1000000.0),
        )
    )
    assert "fnv1a=UNKNOWN" in old_style, "the field is present and explicitly unanswered"

    with pytest.raises(FB.BatchError, match="not an integer"):
        _verdict(old_style + "\n", _policy_for(oracle, value_key="fnv1a"))


# --------------------------------------------------------------- MUTATION 2: an undeclared group
def test_a_group_the_program_printed_and_the_policy_never_declared_fails_admission() -> None:
    """An unchecked group is not a passing one.

    The policy declares three of the four groups.  Every declared checksum matches, every marker is
    present, and the window published a cycle count -- so under the rule that walked only the
    declared set this window passed with a quarter of the model unverified.
    """
    harness, model, emulation, oracle = _oracle()
    partial = {group: values for group, values in oracle.items() if group != "4"}
    policy = _policy_for(partial, value_key="fnv1a")
    uart = _uart(harness, model, emulation, oracle)

    verdict = _verdict(uart, policy)

    assert verdict.status == FB.FAIL
    assert verdict.undeclared_groups == ("4",)
    assert verdict.checksum_mismatches == (), "nothing declared was wrong; something undeclared was unchecked"
    assert "never checked" in verdict.reason
    assert verdict.to_dict()["undeclared_groups"] == ["4"]


def test_a_declared_group_the_window_never_printed_is_absent_and_not_a_sentinel() -> None:
    """The ``-1`` collision, removed.

    A checksum is signed, so ``-1`` is a value a correct run can publish.  With ``-1`` standing for
    "no line", a policy legitimately expecting ``-1`` was satisfied by a window that printed nothing
    for that group at all.  Absence is now its own thing, and that policy is refused.
    """
    harness, model, emulation, oracle = _oracle()
    expecting_minus_one = {group: dict(values) for group, values in oracle.items()}
    expecting_minus_one["3"]["fnv1a"] = -1
    policy = _policy_for(expecting_minus_one, value_key="fnv1a")
    without_group_3 = "\n".join(
        line for line in _uart(harness, model, emulation, oracle).splitlines() if not line.startswith("GM_GROUP 3 ")
    )

    verdict = _verdict(without_group_3 + "\n", policy)

    assert verdict.status == FB.FAIL
    assert verdict.checksum_mismatches == (("3", None, -1),), "absent, not -1"
    assert "ABSENT" in verdict.reason
    row = next(row for row in verdict.to_dict()["checksum_mismatches"] if row["group"] == "3")
    assert row == {"group": "3", "observed": None, "expected": -1, "absent": True}


# --------------------------------------------------------------- MUTATION 3: a duplicated key
def test_a_duplicated_key_raises_in_both_halves_of_the_parser() -> None:
    """One protocol, one reading.  The two helpers used to disagree about this exact case."""
    with pytest.raises(FB.BatchError, match="duplicated key is corruption"):
        FB._key_values(["label=a", "label=b"])
    with pytest.raises(FireSimReceiptError, match="duplicate"):
        _key_values(["label=a", "label=b"], role="queue client", line_number=1)


def test_a_window_frame_with_a_duplicated_label_is_refused_rather_than_resolved() -> None:
    """The consequence: a frame marker naming two labels no longer silently frames the first."""
    harness, model, emulation, oracle = _oracle()
    uart = _uart(harness, model, emulation, oracle).replace(
        "MERLIN_WINDOW begin label=group_model", "MERLIN_WINDOW begin label=group_model label=something_else"
    )

    with pytest.raises(FB.BatchError, match="duplicated key is corruption"):
        _verdict(uart, _policy_for(oracle, value_key="fnv1a"))


def test_console_noise_that_is_not_a_protocol_line_is_still_not_parsed() -> None:
    """The strictness is on the protocol, not on the boot log.

    ``===`` and a trailing ``batch=`` both occur in the vendored queue logs.  A window must still be
    framed in a log that carries them, or this rule would refuse real runs for the console's sake.
    """
    harness, model, emulation, oracle = _oracle()
    noisy = "=== boot === boot ===\n# trailing batch= field\n" + _uart(harness, model, emulation, oracle)

    assert _verdict(noisy, _policy_for(oracle, value_key="fnv1a")).status == FB.PASS


# --------------------------------------------------------------- MUTATION 4: incommensurable windows
def test_a_ratio_across_two_window_kinds_is_refused() -> None:
    """A contiguous wall window over a sum of per-call deltas is not a speedup."""
    contiguous = FB.MeasuredWindow("layer_bench_gemm", 1_000_000, FB.WINDOW_KIND_CONTIGUOUS)
    summed = FB.MeasuredWindow("group_model", 25_400_000, FB.WINDOW_KIND_SUM_OF_CALLS)

    with pytest.raises(FB.BatchError, match="not a speedup"):
        FB.cycle_ratio(summed, contiguous)
    with pytest.raises(FB.BatchError, match="not a speedup"):
        FB.cycle_ratio(contiguous, summed)


def test_a_ratio_within_one_window_kind_is_allowed() -> None:
    first = FB.MeasuredWindow("a", 2_000_000, FB.WINDOW_KIND_CONTIGUOUS)
    second = FB.MeasuredWindow("b", 1_000_000, FB.WINDOW_KIND_CONTIGUOUS)

    assert FB.cycle_ratio(first, second) == pytest.approx(2.0)


def test_an_undeclared_window_kind_is_refused_and_never_assumed_contiguous() -> None:
    """Defaulting the undeclared side is how the incommensurable ratio was taken the first time."""
    undeclared = FB.MeasuredWindow("legacy", 25_400_000)
    known = FB.MeasuredWindow("layer_bench_gemm", 1_000_000, FB.WINDOW_KIND_CONTIGUOUS)

    assert undeclared.declared_kind == FB.WINDOW_KIND_UNKNOWN
    with pytest.raises(FB.BatchError, match="declares no window_kind"):
        FB.cycle_ratio(undeclared, known)


def test_a_window_kind_is_carried_from_the_policy_into_the_sealed_receipt() -> None:
    from merlin.perf.firesim_receipt import CycleClaim

    policy = _policy_for(_oracle()[3], value_key="fnv1a")
    window = policy.window("group_model")

    claim = CycleClaim(
        window=window.label,
        group_checksums=window.checksums,
        checksum_line=tuple(sorted(policy.checksum_line.to_dict().items())),
        window_kind=window.window_kind,
    )

    assert claim.to_dict()["window_kind"] == FB.WINDOW_KIND_SUM_OF_CALLS
    assert CycleClaim(window="w", group_checksums=(), checksum_line=()).to_dict()["window_kind"] == "UNKNOWN"


def test_an_unknown_window_kind_string_is_refused_at_the_policy() -> None:
    with pytest.raises(FB.BatchError, match="window_kind must be one of"):
        FB.WindowPolicy(label="w", markers=(), checksums=(("1", 1),), window_kind="whole_program_wall")


def test_a_policy_without_a_window_kind_serialises_exactly_as_it_did() -> None:
    """So a receipt sealed against a document written before this field keeps its digest."""
    document = json.loads(_RETIRED_POLICY.read_text(encoding="utf-8"))

    assert FB.BatchValidationPolicy.from_json(document).to_dict() == document


# --------------------------------------------------------------- MUTATION 5: the staged binary
def _observe_plan(tmp_path: Path, payload: bytes = b"the elf") -> tuple[FC.CheckpointPlan, Path]:
    queue = tmp_path / "install" / "bin" / "firesim-queue"
    queue.parent.mkdir(parents=True)
    queue.write_text("#!/bin/sh\n", encoding="utf-8")
    queue.chmod(0o755)
    for directory in ("state", "chipyard"):
        (tmp_path / directory).mkdir()
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"schema": "x"}), encoding="utf-8")
    (tmp_path / "a.elf").write_bytes(payload)
    document = {
        "schema": FC.MANIFEST_SCHEMA,
        "checkpoint": "synthetic",
        "host": {
            "queue_executable": str(queue),
            "queue_state_root": str(tmp_path / "state"),
            "chipyard": str(tmp_path / "chipyard"),
        },
        "design": {"hw_config": "design_a", "substrate": "firesim_design_a"},
        "workload": {"name": "synthetic-workload", "bootbinary": "synthetic.elf"},
        "entries": [
            {
                "label": "a",
                "model": "synthetic_model",
                "experiment": "checkpoint",
                "elf": str(tmp_path / "a.elf"),
                "validation_policy": str(policy),
                "observe": {
                    "cycles_prefix": "METRIC cycles ",
                    "after": "MERLIN_PROFILE measured begin",
                    "before": "MERLIN_PROFILE measured end rc=0",
                },
            }
        ],
    }
    return FC.load_plan(document), tmp_path / "a.elf"


_FINISHED_UART = (
    "\n".join(("MERLIN_PROFILE measured begin", "METRIC cycles 4242", "MERLIN_PROFILE measured end rc=0")) + "\n"
)


def _finished_job(plan: FC.CheckpointPlan, job_id: int = 900) -> None:
    """The queue state a same-user job leaves behind: a record and a UART, and no overlay at all."""
    job = plan.host.queue_state_root / "jobs" / str(job_id)
    (job / "simulation" / "sim_slot_0").mkdir(parents=True)
    (job / "simulation" / "sim_slot_0" / "uartlog").write_text(_FINISHED_UART, encoding="utf-8")
    # The native deploy tree, which is where a same-user job's completed UART lands.
    landed = (
        plan.host.chipyard
        / "sims/firesim/deploy/results-workload"
        / f"2026-{plan.workload}-q{job_id}"
        / f"{plan.workload}0"
        / "uartlog"
    )
    landed.parent.mkdir(parents=True)
    landed.write_text(_FINISHED_UART, encoding="utf-8")
    (job / "runworkload-full.json").write_text(
        json.dumps(
            {
                "hw_config": plan.hw_config,
                "hwdb_config_artifact_sha256": None,
                "bootbinary": plan.bootbinary,
                "workload": plan.workload,
            }
        ),
        encoding="utf-8",
    )


def test_a_same_user_job_is_bound_by_the_submit_time_digest_when_no_overlay_exists(tmp_path: Path) -> None:
    """The structurally dead check, replaced by one that can succeed.

    ``deploy_overlay/workloads/`` exists only for a job submitted by a user other than the daemon's
    own, so for these jobs the glob matched nothing and EVERY job this function was pointed at was
    refused for "not staging this entry's executable" -- whatever it had really run.
    """
    plan, elf = _observe_plan(tmp_path)
    _finished_job(plan)
    ledger = tmp_path / "ledger"
    FC.record_staged_binary(plan.submission_for(plan.entries[0]), 900, FC.sha256_file(elf), ledger=ledger)

    row = FC.observe_finished_job(plan, plan.entries[0], 900, tmp_path / "evidence", ledger=ledger)

    assert row["binary_bound_by"] == FC.BOUND_BY_SUBMIT_RECORD
    assert row["stage_from_sha256"] == FC.sha256_file(elf)
    assert row["observed_cycles"] == 4242
    assert row["status"] == "observed", "the UART is in the finished-job location, and no overlay contradicts it"


def test_a_job_submitted_with_different_bytes_is_refused_as_a_stale_binary(tmp_path: Path) -> None:
    """The mutation: the ELF changed after the job ran.  The recorded digest is what catches it."""
    plan, elf = _observe_plan(tmp_path)
    _finished_job(plan)
    ledger = tmp_path / "ledger"
    FC.record_staged_binary(plan.submission_for(plan.entries[0]), 900, "0" * 64, ledger=ledger)

    with pytest.raises(FC.CheckpointError, match="a stale binary ran"):
        FC.observe_finished_job(plan, plan.entries[0], 900, tmp_path / "evidence", ledger=ledger)
    assert FC.sha256_file(elf) != "0" * 64


def test_with_neither_an_overlay_nor_a_record_the_result_is_not_attributed(tmp_path: Path) -> None:
    """Fail closed: "the entry names an ELF, so that must be what ran" is not evidence."""
    plan, _elf = _observe_plan(tmp_path)
    _finished_job(plan)

    with pytest.raises(FC.CheckpointError, match="cannot be bound to this entry's bytes"):
        FC.observe_finished_job(plan, plan.entries[0], 900, tmp_path / "evidence", ledger=tmp_path / "empty")


def test_a_submission_records_the_digest_of_the_bytes_it_staged(tmp_path: Path) -> None:
    """The submit half: hashed before the queue is asked to run it, and written where it is findable."""
    plan, elf = _observe_plan(tmp_path)
    submission = plan.submission_for(plan.entries[0])
    ledger = tmp_path / "ledger"

    written = FC.record_staged_binary(submission, 901, FC.sha256_file(elf), ledger=ledger)
    read_back = FC.staged_binary_record(901, ledger=ledger)

    assert written.is_file() and read_back is not None
    assert read_back["stage_from_sha256"] == FC.sha256_file(elf)
    assert read_back["stamp"].startswith(f"job=901 elf={FC.sha256_file(elf)[:12]}")
    assert FC.staged_binary_record(902, ledger=ledger) is None
