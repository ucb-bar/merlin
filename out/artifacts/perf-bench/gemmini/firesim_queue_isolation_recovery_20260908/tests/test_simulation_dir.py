#!/usr/bin/env python3
"""Regression for job-private FireSim simulation directories."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
import uuid


QUEUE_MODULE = Path("/scratch/firesim_queue/bin/firesim_queue.py")


class SimulationDirectoryTest(unittest.TestCase):
    def test_renderer_replaces_shared_simulation_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fq-simdir-test-") as raw:
            root = Path(raw)
            old_root = os.environ.get("FIRESIM_QUEUE_ROOT")
            os.environ["FIRESIM_QUEUE_ROOT"] = str(root / "queue")
            try:
                spec = importlib.util.spec_from_file_location(
                    f"firesim_queue_test_{uuid.uuid4().hex}", QUEUE_MODULE
                )
                assert spec is not None and spec.loader is not None
                queue = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(queue)
            finally:
                if old_root is None:
                    os.environ.pop("FIRESIM_QUEUE_ROOT", None)
                else:
                    os.environ["FIRESIM_QUEUE_ROOT"] = old_root

            template = root / "template.yaml"
            output = root / "rendered.yaml"
            private = root / "job-42" / "simulation"
            template.write_text(
                "run_farm:\n"
                "  recipe_arg_overrides:\n"
                "    default_simulation_dir: /shared/stale-state\n"
                "target_config:\n"
                "    default_hw_config: old_hw\n"
                "workload:\n"
                "    workload_name: old.json\n"
                "    suffix_tag: old\n",
                encoding="utf-8",
            )

            queue._render_per_job_runtime_yaml(
                template,
                output,
                "new-workload",
                "q42",
                hw_config="new_hw",
                simulation_dir=str(private),
            )
            rendered = output.read_text(encoding="utf-8")
            self.assertIn(f"default_simulation_dir: {private}", rendered)
            self.assertNotIn("/shared/stale-state", rendered)
            self.assertIn("default_hw_config: new_hw", rendered)
            self.assertIn("workload_name: new-workload.json", rendered)
            self.assertIn("suffix_tag: q42", rendered)


if __name__ == "__main__":
    unittest.main()
