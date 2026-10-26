"""Small own-child controls; system Python only, never real intake."""
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import fdio
import watchdog
import watchdog_runner

HERE = Path(__file__).absolute().parent
OUTPUT = HERE / "synthetic-watchdog-v1"


class WatchdogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUTPUT.mkdir()  # refuse rerun over retained evidence

    def call(self, mode, seconds=.5):
        workspace = OUTPUT / self._testMethodName
        command = [sys.executable, "-I", "-S", "-B", str(HERE / "own_child_control.py"), mode]
        result = watchdog.supervise(command, workspace, seconds=seconds, terminate_seconds=.1, reap_seconds=.2)
        self.assertTrue(result["child_reaped"])
        self.assertTrue((workspace / "PARENT-ATTEMPT.json").is_file())
        self.assertEqual(json.loads((workspace / "OWNED-CHILD.json").read_text())["pid"], result["owned_pid"])
        return result, workspace

    def timeout(self, mode):
        result, path = self.call(mode)
        self.assertEqual(result["status"], "COMPLETE_INTAKE_BUDGET_EXCEEDED")
        self.assertTrue(result["forced_stop"])
        self.assertEqual((path / "CONTROL-STARTED.txt").read_text(), mode)
        self.assertIn("in_memory_evidence", json.loads((path / "FIRST-PARENT-FAILURE.json").read_text()))
        self.assertTrue(all(s["owned_pid"] == result["owned_pid"] for s in result["signals"]))
        return result, path

    def test_normal(self):
        result, path = self.call("normal", 2)
        self.assertEqual(result["status"], "CHILD_COMPLETED")
        self.assertFalse(result["signals"])
        self.assertTrue((path / "COMPLETE.txt").is_file())

    def test_block_open(self):
        self.timeout("open")

    def test_block_read(self):
        _, path = self.timeout("read")
        self.assertEqual((path / "partial.bin").read_bytes(), b"retained before blocked read")

    def test_block_close(self):
        _, path = self.timeout("close")
        self.assertTrue((path / "CLOSE-ENTERED.txt").is_file())
        self.assertEqual((path / "partial.bin").read_bytes(), b"retained before blocking close")

    def test_kill_reap(self):
        result, path = self.timeout("kill")
        self.assertTrue((path / "IGNORE-TERM-READY.txt").is_file())
        self.assertEqual([s["signal"] for s in result["signals"]], ["SIGTERM", "SIGKILL"])

    def test_nonzero(self):
        result, path = self.call("nonzero", 2)
        self.assertEqual(result["status"], "CHILD_FAILED")
        self.assertEqual(result["child_exit_code"], 7)
        self.assertTrue((path / "FIRST-PARENT-FAILURE.json").is_file())

    def test_no_resume(self):
        _, path = self.call("normal", 2)
        before = (path / "PARENT-ATTEMPT.json").read_bytes()
        with patch.object(watchdog.subprocess, "Popen", side_effect=AssertionError("must not relaunch")):
            with self.assertRaises(fdio.IntakeStop):
                watchdog.supervise([sys.executable, "-I", "-S", "-B", str(HERE / "own_child_control.py"), "normal"],
                                   path, seconds=1, terminate_seconds=.1, reap_seconds=.1)
        self.assertEqual(before, (path / "PARENT-ATTEMPT.json").read_bytes())

    def test_invalid_command_and_grace(self):
        command = [sys.executable, "-I", "-S", "-B", str(HERE / "own_child_control.py"), "normal"]
        with patch.object(watchdog.subprocess, "Popen", side_effect=AssertionError("no launch")):
            for cmd, term, reap in ((["kill", "123"], 1, 1), (command, 3, 3), (command, float("nan"), 1)):
                with self.assertRaises(ValueError):
                    watchdog.supervise(cmd, OUTPUT / "invalid", seconds=1, terminate_seconds=term, reap_seconds=reap)

    def test_parent_prelaunch_failure(self):
        command = [sys.executable, "-I", "-S", "-B", str(HERE / "own_child_control.py"), "normal"]
        path = OUTPUT / self._testMethodName
        with patch.object(watchdog.subprocess, "Popen", side_effect=OSError("injected launch failure")):
            with self.assertRaises(fdio.IntakeStop):
                watchdog.supervise(command, path, seconds=1, terminate_seconds=.1, reap_seconds=.1)
        self.assertEqual(json.loads((path / "FIRST-PARENT-FAILURE.json").read_text())["error"], "injected launch failure")

    def test_wrong_source_pin(self):
        raw = (HERE / "own_child_control.py").read_bytes()
        with self.assertRaises(ValueError):
            watchdog_runner.bound({"path": str(HERE / "own_child_control.py"), "bytes": len(raw), "sha256": "0" * 64})

    def test_proposal_not_permission(self):
        document = {"schema": "merlin.own-child-intake-watchdog-review.v1", "status": "PROPOSED_NOT_AUTHORIZED",
                    "action": "ONE_COMPLETE_DATA_INTAKE_CHILD", "attempts": 1, "source_refs": [],
                    "child_authorization": {}, "workspace": str(OUTPUT / "unauthorized"), "seconds": 1800,
                    "terminate_seconds": 2, "reap_seconds": 3, "native_authorized": False,
                    "install_authorized": False, "old_holds_cleared": False}
        raw = json.dumps(document).encode()
        path = OUTPUT / "proposal.json"
        path.write_bytes(raw)
        with patch.object(sys, "argv", ["runner", "--authorization", str(path), "--authorization-sha256", hashlib.sha256(raw).hexdigest()]):
            with self.assertRaisesRegex(ValueError, "not complete-intake permission"):
                watchdog_runner.main()


if __name__ == "__main__":
    unittest.main(verbosity=2)
