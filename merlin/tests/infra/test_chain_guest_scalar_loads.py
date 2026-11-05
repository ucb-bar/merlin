"""Exercise the real guest's scalar readers under its freestanding policy."""

import shutil
import subprocess
from pathlib import Path


def test_scalar_readers_preserve_unaligned_storage_and_ieee_values(tmp_path):
    runtime = Path(__file__).resolve().parents[3] / "merlin/runtime/c"
    (tmp_path / "chain_data.h").write_text("""
#define CHAIN_PROGRAMS 0
#define CHAIN_STEPS 0
#define CHAIN_MAX_ARGS 1
#define CHAIN_COMPACT 1
#define CHAIN_SAMPLES 4
#define CHAIN_ATOL 0.000002
#define CHAIN_RTOL 0.00001
static const chain_program PROGRAMS[1] = {{0}};
static const chain_step STEPS[1] = {{0}};
static const unsigned char CHAIN_INPUTS[1], CHAIN_REFERENCES[1];
static uint64_t chain_counter_now(void) { return 0; }
""")
    source = tmp_path / "readers.c"
    source.write_text("""
#define MERLIN_CHAIN_HOST_FIXTURE
#define main retained_chain_main
#include "chain_guest.c"
#undef main
int main(void) {
    unsigned char data[16];
    for (int i = 0; i < 16; ++i) data[i] = (unsigned char)(17*i + 3);
    for (int offset = 0; offset < 8; ++offset) {
        for (int width = 1; width <= 8; ++width) {
            uint64_t expected = 0;
            for (int i = 0; i < width; ++i)
                expected |= (uint64_t)data[offset+i] << (8*i);
            if (bits(data + offset, width) != expected) return 1;
        }
    }
    const uint64_t one_and_half[] = {0, 0x3fc00000, 0x3ff8000000000000ULL, 0x3fc0, 0x3e00};
    const uint64_t infinity[] = {0, 0x7f800000, 0x7ff0000000000000ULL, 0x7f80, 0x7c00};
    const uint64_t negative_zero[] = {0, 0x80000000, 0x8000000000000000ULL, 0x8000, 0x8000};
    for (int kind = 1; kind <= 4; ++kind) {
        int finite = 1;
        if (real_value(one_and_half[kind], kind, &finite) != 1.5 || !finite) return 2;
        (void)real_value(infinity[kind], kind, &finite);
        if (finite) return 3;
        finite = 1;
        (void)real_value(infinity[kind] | 1, kind, &finite);
        if (finite) return 4;
        finite = 1;
        double value = real_value(negative_zero[kind], kind, &finite);
        if (!finite || bits(&value, 8) != 0x8000000000000000ULL) return 5;
    }
    return 0;
}
""")
    executable = tmp_path / "readers"
    result = subprocess.run(
        [
            shutil.which("gcc"),
            "-O2",
            "-ffreestanding",
            "-fno-builtin",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-ffunction-sections",
            "-fdata-sections",
            "-Wl,--gc-sections",
            "-I" + str(runtime),
            "-I" + str(tmp_path),
            str(source),
            "-o",
            str(executable),
        ],
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr.decode()
    subprocess.run([str(executable)], check=True, capture_output=True, timeout=5)
