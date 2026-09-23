"""The N-window program's UART must be the one the FireSim receipt parser accepts.

This is the gap the harness fell into once already: the layer-bench programs printed a correct
measurement in a spelling ``merlin.perf.firesim_receipt`` could not read, so no run of them could
ever have been sealed -- independently of whether the arithmetic was right. A comment cannot detect
that; only holding the emitted lines against the parser can, which is what these tests do.

The expected UART is rendered from the program's OWN spelling table
(:data:`~merlin.targets.gemmini.backend.gemmini_layer_bench.WINDOW_UART`), never hand-written: a
test that retyped the lines would grade the harness against the same guess that wrote it.
"""

from __future__ import annotations

import pytest

from merlin.perf.firesim_batch import (
    FAIL,
    INCOMPLETE,
    PASS,
    BatchMember,
    BatchValidationPolicy,
    admit_batch,
    admit_window,
    link_batch,
)
from merlin.perf.firesim_receipt import _verify_uart
from merlin.runtime.backends import base

TARGET = "gemmini"
DIGESTS = {"w0": 8466922728085698976, "w1": 1234567890123456789}


def _bench():
    return base.get_backend(TARGET)


def _windows(labels):
    """Window descriptors for a plain library matmul each, with disjoint operand offsets."""
    return [
        {
            "label": label,
            "spec": {"op": "matmul", "m": 64, "n": 64, "k": 64, "relu": False, "scale": 0.0009765625},
            "offsets": {"a": 0, "b": 4096, "d": 8192},
        }
        for label in labels
    ]


def _frame_lines(uart: dict, label: str, cycles: int, digest: int) -> list[str]:
    return [
        uart["window_begin"].format(label=label),
        uart["warm_begin"],
        uart["warm_end"],
        uart["measured_begin"],
        uart["record"].format(label=label, cycles=cycles, digest=digest),
        uart["done"].format(label=label),
        uart["metric"].format(cycles=cycles),
        uart["measured_end"],
        uart["window_end"].format(label=label),
    ]


def _policy(labels, digests, workload="merlin-checkpoint") -> BatchValidationPolicy:
    bench = _bench()
    return BatchValidationPolicy.from_json(
        {
            "schema": "merlin_firesim_uart_validation_policy_v2",
            "policy_id": "test/window-program",
            "workload": workload,
            "checksum_line": dict(bench.WINDOW_CHECKSUM_LINE),
            "per_window": [
                {"label": label, "markers": [bench.window_marker(label)], "checksums": {label: digests[label]}}
                for label in labels
            ],
        }
    )


def test_every_protocol_line_the_program_prints_is_a_literal_in_its_source():
    """The C really emits the spelling the policy and the parser are built from.

    Checked as literals rather than by running the program: the point is that the two renderers --
    the C ``printf`` and the expected-UART table -- cannot drift apart, and a drift shows up here
    before it shows up as a queue slot spent on an unsealable log.
    """
    bench = _bench()
    source = bench.render_window_program(_windows(["w0"]), batch_id=None)
    uart = bench.WINDOW_UART
    for line in [uart["invocations"], *_frame_lines(uart, "w0", "%llu", "%llu")]:
        assert line in source, line
    # A solo program must NOT open a batch: a batch frame declaring one window would make the
    # UART a batch log that the solo receipt path has no plan to check it against.
    assert "MERLIN_BATCH" not in source
    assert "MERLIN_BATCH begin" in bench.render_window_program(_windows(["w0"]), batch_id="b1")


def test_a_solo_window_frame_satisfies_the_receipt_parsers_shape_rule():
    """One window's UART is exactly what ``_verify_uart`` admits -- one warm, one measured, one metric."""
    bench = _bench()
    uart = bench.WINDOW_UART
    text = "\n".join([uart["invocations"], *_frame_lines(uart, "w0", 531318, DIGESTS["w0"])])
    _invocation, _profiles, _metric_line, cycles, markers = _verify_uart(text, (bench.window_marker("w0"),))
    assert cycles == 531318
    assert len(markers) == 1


def test_a_batched_uart_admits_every_window_and_its_order_effect_control():
    bench = _bench()
    uart = bench.WINDOW_UART
    labels = ["w0", "w1"]
    control = "w0__order_control"
    digests = {**DIGESTS, control: DIGESTS["w0"]}
    lines = [uart["invocations"], uart["batch_begin"].format(batch="b1", windows=3)]
    for label, cycles in ((labels[0], 531318), (labels[1], 544803), (control, 531400)):
        lines += _frame_lines(uart, label, cycles, digests[label])
    lines.append(uart["batch_end"].format(batch="b1"))
    text = "\n".join(lines) + "\n"

    members = [
        BatchMember(label=label, program_sha256="a" * 64, weights_sha256="b" * 64, observed_window_seconds=5.0)
        for label in labels
    ]
    batch = link_batch(members, batch_id="b1", queue_wall_limit_seconds=1800.0)
    # link_batch MINTS the control's label; the program must print that same one or the window the
    # plan calls the control is not the window the run produced.
    assert batch.repeat_label == control
    admission = admit_batch(text, batch, _policy([*labels, control], digests))
    assert admission.status == PASS, admission.reason
    assert [window.cycles for window in admission.windows] == [531318, 544803, 531400]


def test_a_window_that_ran_and_was_wrong_is_fail_not_incomplete():
    """THE JOB-730 RULE on this harness's own spelling.

    That run cleared every marker it declared and was wrong.  A window here prints its marker and a
    digest; flipping only the digest must move the verdict to ``fail`` -- a wrong result reported as
    an absence is how a wrong number gets cited.
    """
    bench = _bench()
    uart = bench.WINDOW_UART
    text = "\n".join(_frame_lines(uart, "w0", 531318, DIGESTS["w0"] ^ 1))
    policy = _policy(["w0"], DIGESTS)
    admission = admit_window(text, policy.window("w0"), policy.checksum_line)
    assert admission.status == FAIL
    assert admission.cycles == 531318
    assert admission.checksum_mismatches


def test_a_window_that_never_ran_is_incomplete_and_stays_in_the_denominator():
    bench = _bench()
    policy = _policy(["w0"], DIGESTS)
    admission = admit_window(bench.WINDOW_UART["invocations"], policy.window("w0"), policy.checksum_line)
    assert admission.status == INCOMPLETE
    assert admission.cycles is None


def test_window_labels_must_be_unique_and_frameable():
    bench = _bench()
    with pytest.raises(ValueError):
        bench.render_window_program(_windows(["w0", "w0"]), batch_id="b1")
    with pytest.raises(ValueError):
        bench.render_window_program(_windows(["has space"]), batch_id="b1")
    with pytest.raises(ValueError):
        bench.render_window_program(_windows(["has=equals"]), batch_id="b1")


def _pinned_job734() -> str:
    """The real FireSim job-734 UART, comment lines dropped."""
    from merlin.common.paths import merlin_dir

    path = merlin_dir() / "tests" / "data" / "firesim_queue" / "job734_gemm_batch_uart_skeleton.log"
    return "\n".join(line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#"))


def test_the_batched_frame_is_pinned_to_the_log_hardware_actually_printed():
    """The admission rules must hold against the run, not against a shape written to match them.

    ``firesim_batch`` could link and admit a batch before anything had ever printed one -- its own
    docstring said the frame was "PROPOSED, NOT OBSERVED". This pins it to FireSim job 734, where 13
    windows ran in one ``runworkload-full``, so a change to the frame that the hardware would not
    produce fails here instead of on the next queue slot.
    """
    text = _pinned_job734()
    labels = [
        line.split()[2].partition("=")[2] for line in text.splitlines() if line.startswith("MERLIN_WINDOW begin ")
    ]
    assert len(labels) == 13
    assert labels[-1] == labels[0] + "__order_control", "the last window must repeat window 0"

    digests = {}
    for line in text.splitlines():
        if line.startswith("LB_RECORD "):
            tokens = line.split()
            digests[tokens[1]] = int(dict(t.partition("=")[::2] for t in tokens[2:])["digest"])
    policy = _policy(labels, digests)
    members = [
        BatchMember(label=label, program_sha256="a" * 64, weights_sha256="b" * 64, observed_window_seconds=13.4)
        for label in labels[:-1]
    ]
    batch = link_batch(members, batch_id="gemm6x2", queue_wall_limit_seconds=1800.0)
    admission = admit_batch(text, batch, policy)
    assert admission.status == PASS, admission.reason
    assert admission.order_effect_ppm == 26
    assert all(window.status == PASS for window in admission.windows)


def test_the_pinned_run_is_refused_when_one_window_computed_something_else():
    """Flip one digest of the REAL log: that window becomes fail, and the batch stops being usable."""
    text = _pinned_job734()
    bad = text.replace("digest=8355844006066967625", "digest=8355844006066967624", 1)
    assert bad != text
    labels = [
        line.split()[2].partition("=")[2] for line in text.splitlines() if line.startswith("MERLIN_WINDOW begin ")
    ]
    digests = {}
    for line in text.splitlines():
        if line.startswith("LB_RECORD "):
            tokens = line.split()
            digests[tokens[1]] = int(dict(t.partition("=")[::2] for t in tokens[2:])["digest"])
    policy = _policy(labels, digests)
    members = [
        BatchMember(label=label, program_sha256="a" * 64, weights_sha256="b" * 64, observed_window_seconds=13.4)
        for label in labels[:-1]
    ]
    batch = link_batch(members, batch_id="gemm6x2", queue_wall_limit_seconds=1800.0)
    admission = admit_batch(bad, batch, policy)
    wrong = [window for window in admission.windows if window.status == FAIL]
    assert len(wrong) == 1
    assert wrong[0].label == "s12544x256x64"
    assert wrong[0].cycles == 971097, "a wrong window still reports what it measured, as a FAILURE"


# --------------------------------------------------------------- external-source windows


PUBLISHED_KERNEL = (
    "void solution(int8_t A[64][32], int8_t B[32][16], int8_t C[64][16]) {\n"
    "  config_st((16));\n"
    "  mvin(&A[0][0], 0, 16, 16);\n"
    "  fence();\n"
    "}\n"
)


def test_a_published_kernel_is_carried_into_its_unit_byte_for_byte():
    """The bench may supply context around a third party's kernel; it may not rewrite a line of it.

    A bench that edited the kernel to make its numbers land would be measuring its own edit, so the
    published text has to appear in the compiled unit exactly as it was read.
    """
    unit = _bench().render_autocomp_kernel_unit(
        PUBLISHED_KERNEL, symbol="lb_pub_0", m=64, n=16, k=32, declared_shape={"m": 64, "n": 16, "k": 32}
    )
    assert PUBLISHED_KERNEL.rstrip() in unit
    # The rename is a preprocessor definition, never a substitution inside the text.
    assert "#define solution lb_pub_0" in unit
    assert unit.count("void solution(") == 1
    # The extents the unit compiles against come from the measured spec, not from the file.
    assert "#define MAT_DIM_I 64" in unit and "#define MAT_DIM_J 16" in unit and "#define MAT_DIM_K 32" in unit


def test_a_published_kernel_is_refused_at_extents_it_was_not_written_for():
    """These kernels carry their tiling as literals, so they are not themselves on another shape."""
    with pytest.raises(ValueError, match="not that kernel on other extents"):
        _bench().render_autocomp_kernel_unit(
            PUBLISHED_KERNEL, symbol="lb_pub_0", m=64, n=16, k=32, declared_shape={"m": 64, "n": 16, "k": 64}
        )


def test_two_published_windows_of_one_program_call_different_symbols():
    """Every published file names its function ``solution``; one ELF holding six of them cannot.

    The window program must therefore call each shape's kernel by a name of its own, declared in the
    program and defined in that kernel's own unit.
    """
    backend = _bench()
    windows = []
    for index, (shape, offset) in enumerate((((64, 16, 32), 0), ((32, 16, 32), 8192))):
        m, n, k = shape
        declaration, body, n_out, call = backend.autocomp_window_call(
            symbol=f"lb_pub_{index}", offsets={"a": offset, "b": offset + 4096}, m=m, n=n, k=k
        )
        windows.append(
            {
                "label": f"p{index}",
                "spec": {"op": "matmul", "m": m, "n": n, "k": k, "relu": False, "scale": 1.0},
                "offsets": {"a": offset, "b": offset + 4096},
                "kernel_c": declaration,
                "external": {"body": body, "n_out": n_out, "call": call},
            }
        )
    source = backend.render_window_program(windows, batch_id="b")
    assert "lb_pub_0(A, B, C);" in source and "lb_pub_1(A, B, C);" in source
    assert "extern void lb_pub_0(" in source and "extern void lb_pub_1(" in source
    # Each window reads its own operands out of the one shared blob and writes the shared output.
    assert "lb_operands + 0)" in source and "lb_operands + 8192)" in source


def test_an_external_window_must_declare_the_output_the_frame_digests():
    """The frame digests ``output``; a window that never defines it would digest another window's.

    Silently digesting a stale buffer is the failure mode the admission rule exists to catch, so it
    is refused where the program is rendered rather than discovered in a UART log.
    """
    backend = _bench()
    window = {
        "label": "x0",
        "spec": {"op": "matmul", "m": 4, "n": 4, "k": 4, "relu": False, "scale": 1.0},
        "offsets": {"a": 0, "b": 64, "d": 128},
        "kernel_c": "extern void k(void *);\n",
        "external": {"body": "    void *p = 0;\n", "n_out": "16", "call": "k(p);"},
    }
    with pytest.raises(ValueError, match="must define the elem_t \\*output"):
        backend.render_window_program([window])


def test_every_window_poisons_its_output_before_it_runs():
    """Windows share one output buffer, so a partial writer must not inherit a correct digest.

    Without this, the second window of a shape could write nothing and still print the digest the
    first one left behind -- a cycle count admitted for work it did not do. The poison is emitted
    BEFORE the measurement opens, so it costs the number nothing.
    """
    backend = _bench()
    source = backend.render_window_program(_windows(["w0", "w1"]), batch_id="b")
    assert source.count("lb_poison(output,") == 2
    for label in ("w0", "w1"):
        head, _, rest = source.partition(f'printf("MERLIN_WINDOW begin label={label}')
        assert "lb_poison(output," not in head.rsplit("static void lb_window_", 1)[-1]
        poison = rest.index("lb_poison(output,")
        assert poison < rest.index("read_cycles()"), "the poison must land before the measurement opens"
