"""Exercise exact byte diagnostics, alignment, tails and inaccessible neighbors."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).absolute().parents[1]


class HostBitsTest(unittest.TestCase):
    def test_bf16_bit_conversions(self):
        self._run_checker("host-bits-test.c")

    def test_freestanding_bit_conversions_have_no_memcpy(self):
        cc = os.environ.get("RISCV_CC") or shutil.which("riscv64-unknown-elf-gcc")
        if cc is None:
            self.skipTest("RV64 C compiler unavailable")
        nm = str(Path(cc).with_name("riscv64-unknown-elf-nm"))
        with tempfile.TemporaryDirectory(prefix="atlas-host-bits-") as directory:
            obj = Path(directory) / "bits.o"
            subprocess.run([cc, "-std=c11", "-O2", "-ffreestanding", "-fno-builtin", "-fstrict-aliasing",
                            "-march=rv64gc", "-mabi=lp64d", "-I" + str(ROOT / "runtime"),
                            "-c", str(ROOT / "test/host-bits-object.c"), "-o", str(obj)], check=True, timeout=30)
            undefined = subprocess.check_output([nm, "-u", str(obj)], text=True, timeout=30)
            self.assertEqual(undefined.strip(), "", undefined)

    def _run_checker(self, source):
        cc = shutil.which("cc")
        if cc is None:
            self.skipTest("native C compiler unavailable")
        with tempfile.TemporaryDirectory(prefix="atlas-host-check-") as directory:
            executable = Path(directory) / "check"
            subprocess.run([cc, "-std=gnu11", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-fstrict-aliasing", "-fsanitize=undefined", "-fsanitize-undefined-trap-on-error",
                            "-I" + str(ROOT / "runtime"), str(ROOT / "test" / source),
                            "-o", str(executable)], check=True, timeout=30)
            subprocess.run([str(executable)], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
