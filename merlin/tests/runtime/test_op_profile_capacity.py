"""Complete table and sentinel storage for large operation profiles."""

import subprocess

import pytest

from merlin.common.paths import runtime_dir
from merlin.llvmlower.op_profile_capacity import runtime_capacity


@pytest.mark.parametrize("count,expected", [(3, 8192), (8191, 8192), (8192, 8193), (38382, 38383)])
def test_capacity_includes_every_operation_and_closing_sentinel(count, expected):
    assert runtime_capacity([{"id": i} for i in range(count)]) == expected


@pytest.mark.parametrize("table", [[], [{"id": 1}], [{"id": False}], [{"id": 0}, {"id": 0}]])
def test_malformed_tables_refuse(table):
    with pytest.raises(ValueError):
        runtime_capacity(table)


def test_real_runtime_records_ids_above_default_capacity(tmp_path):
    from merlin.runtime.backends import spike

    if not spike.available():
        pytest.skip("selected bare-metal CPU simulator/toolchain unavailable")
    compiler = spike.gcc_path()
    harness = runtime_dir() / "baremetal/spike"
    count = 8201
    capacity = runtime_capacity([{"id": i} for i in range(count)])
    source, executable = tmp_path / "main.c", tmp_path / "profile"
    source.write_text(
        '#include "htif.h"\nvoid merlin_prof_mark(int); void merlin_prof_dump(void);\n'
        f"int main(int hart) {{if(hart) return 0;for(int i=0;i<={count};++i) merlin_prof_mark(i);"
        "merlin_prof_dump();htif_exit(0);return 0;}\n"
    )
    subprocess.run(
        [
            str(compiler),
            "-O2",
            "-march=rv64gc",
            "-mabi=lp64d",
            "-mcmodel=medany",
            "-ffreestanding",
            "-fno-builtin",
            "-nostdlib",
            "-nostartfiles",
            "-static",
            "-DMERLIN_PROF_BAREMETAL",
            "-I" + str(harness),
            "-T" + str(harness / "link.ld"),
            f"-DMERLIN_PROF_MAX_OPS={capacity}",
            str(source),
            str(runtime_dir() / "c/merlin_op_prof.c"),
            str(harness / "crt.S"),
            str(harness / "htif.c"),
            str(harness / "libc_min.c"),
            "-lgcc",
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
    )
    result = subprocess.run(
        [str(spike.spike_path()), "--isa=rv64gc", "--priv=m", "-m128", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    rows = [line.split() for line in (result.stdout + result.stderr).splitlines() if line.startswith("PROF ")]
    assert [int(row[1]) for row in rows] == list(range(count))
    assert all(int(row[3]) == 1 for row in rows)
