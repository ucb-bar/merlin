"""Reject successful bench subprocesses whose numerical receipts fail admission."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

FULL_MODEL = Path(__file__).resolve().parents[1] / "full_model"
sys.path.insert(0, str(FULL_MODEL))
from bench_admission import admit_bench


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, FULL_MODEL / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def qualified_rows():
    return [dict(key=f"variant-{i}", family="projection_gemm" if i < 5 else "attention",
                 status="ok", timed_output_equal=True, bit_exact=True) for i in range(9)]


class BenchAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = qualified_rows()

    def receipt(self, rows=None):
        (self.root / "MODEL-BENCH.json").write_text(json.dumps({"launches": self.rows if rows is None else rows}))

    def test_valid_qualified_receipt_is_admitted(self):
        self.receipt()
        result = admit_bench(self.root, expected_variants=9)
        self.assertEqual(result["status"], "admitted")
        self.assertEqual(result["variants"], 9)

    def test_timed_mismatch_and_missing_or_nonboolean_equality_fail(self):
        for value in (False, None, "true", 1):
            with self.subTest(value=value):
                rows = qualified_rows()
                rows[-1]["timed_output_equal"] = value
                self.receipt(rows)
                with self.assertRaisesRegex(RuntimeError, "qualification failed"):
                    admit_bench(self.root)
        del rows[-1]["timed_output_equal"]
        self.receipt(rows)
        with self.assertRaises(RuntimeError):
            admit_bench(self.root)

    def test_projection_arithmetic_mismatch_or_missing_field_fails(self):
        for value in (False, None, "true", 1):
            with self.subTest(value=value):
                rows = qualified_rows()
                rows[0]["bit_exact"] = value
                self.receipt(rows)
                with self.assertRaises(RuntimeError):
                    admit_bench(self.root)
        del rows[0]["bit_exact"]
        self.receipt(rows)
        with self.assertRaises(RuntimeError):
            admit_bench(self.root)

    def test_failed_empty_duplicate_or_wrong_census_receipts_fail(self):
        failed = qualified_rows()
        failed[0]["status"] = "failed"
        for rows in (failed, [], [self.rows[0]] * 9, self.rows[:-1]):
            with self.subTest(rows=len(rows)):
                self.receipt(rows)
                with self.assertRaises(RuntimeError):
                    admit_bench(self.root, expected_variants=9)

    def test_nonprojection_does_not_require_independent_golden(self):
        self.rows[-1].pop("bit_exact")
        self.receipt()
        self.assertEqual(admit_bench(self.root)["status"], "admitted")

    def run_driver(self, filename, rows, plan_only=False):
        driver = load("qwen_admission_" + filename.replace(".", "_"), filename)
        output = self.root / filename.replace(".", "_")
        library, model_root, atlas_bin = (self.root / p for p in ("library", "model", "bin"))
        for path in (library / "coverage/export_ops.py", model_root / "npu_model/simulation.py",
                     atlas_bin / "atlas-opt", atlas_bin / "atlas-emit"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        calls = []

        def execute(command, *args, **kwargs):
            calls.append(list(map(str, command)))
            if any(str(item).endswith("bench_model.py") for item in command) and not plan_only:
                (output / "bench").mkdir()
                (output / "bench/MODEL-BENCH.json").write_text(json.dumps({"launches": rows}))

        common = ["--out", str(output), "--atlas-bin", str(atlas_bin),
                  "--model-root", str(model_root)] + (["--plan-only"] if plan_only else [])
        if filename == "run.py":
            argv = [filename, *common, "--library-root", str(library), "--capture-python", sys.executable,
                    "--model-python", sys.executable, "--skip-real-capture"]
            toolchain = types.SimpleNamespace(bin_dir=lambda: atlas_bin)
            with patch.dict(sys.modules, {"atlas_mlir.toolchain": toolchain}), \
                 patch.object(driver, "bootstrap_compiler", return_value=self.root), \
                 patch.object(driver, "authenticate", return_value={"checkpoint_authenticated": False}), patch.object(driver, "observed_revision", return_value="local"), \
                 patch.object(driver, "run", side_effect=execute), patch.object(sys, "argv", argv):
                driver.main()
        else:
            argv = [filename, *common, "--library", str(library)]
            with patch.object(driver, "annotate"), patch.object(driver, "run", side_effect=execute), \
                 patch.object(driver.subprocess, "check_output", return_value="local\n"), \
                 patch.object(sys, "argv", argv):
                driver.main()
        return output, calls

    def test_both_drivers_reject_bad_receipts_before_later_work(self):
        for filename in ("run.py", "launch_bench.py"):
            for field in ("timed_output_equal", "bit_exact"):
                with self.subTest(driver=filename, field=field):
                    rows = qualified_rows()
                    rows[0][field] = False
                    with self.assertRaisesRegex(RuntimeError, "qualification failed"):
                        self.run_driver(filename, rows)
                    output = self.root / filename.replace(".", "_")
                    self.assertFalse((output / "run.json").exists())
                    self.assertFalse((output / "capture").exists())
                    self.assertFalse((output / "qualification").exists())
                    import shutil
                    shutil.rmtree(output)

    def test_both_drivers_plan_only_never_require_or_claim_qualification(self):
        for filename in ("run.py", "launch_bench.py"):
            with self.subTest(driver=filename):
                output, calls = self.run_driver(filename, None, plan_only=True)
                receipt = json.loads((output / "run.json").read_text())
                self.assertEqual(receipt["launch_qualification"]["status"], "PENDING")
                self.assertFalse(any(any("capture_boundaries.py" in item for item in call) for call in calls))

    def test_pretrained_driver_accepts_admitted_nonplan_receipt(self):
        output, _ = self.run_driver("run.py", qualified_rows())
        result = json.loads((output / "run.json").read_text())
        self.assertEqual(result["launch_qualification"]["status"], "admitted")


if __name__ == "__main__":
    unittest.main()
