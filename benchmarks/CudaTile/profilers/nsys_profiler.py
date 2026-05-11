"""Nsight Systems profiler: captures GPU kernel times via nsys + SQLite."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class NsysResult:
    median_kernel_us: float = 0.0
    all_kernel_us: list[float] = field(default_factory=list)
    success: bool = False
    error: str = ""


class NsysProfiler:
    def profile_iree(
        self,
        iree_run: str,
        vmfb: Path,
        device: str,
        reps: int = 10,
    ) -> NsysResult:
        with tempfile.NamedTemporaryFile(suffix=".nsys-rep", delete=False) as f:
            report_path = f.name

        cmd = [
            "nsys",
            "profile",
            "--stats=true",
            f"--output={report_path}",
            "--force-overwrite=true",
            iree_run,
            f"--device={device}",
            f"--module={vmfb}",
            f"--benchmark_repetitions={reps}",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                return NsysResult(error=f"nsys failed: {result.stderr[:200]}")

            sqlite_path = self._find_sqlite(report_path)
            if sqlite_path:
                return self._parse_sqlite(sqlite_path)
            return NsysResult(error="no nsys SQLite output found")

        except subprocess.TimeoutExpired:
            return NsysResult(error="nsys timed out")
        finally:
            for ext in ["", ".sqlite", ".nsys-rep"]:
                p = report_path + ext
                if os.path.isfile(p):
                    os.unlink(p)

    def _find_sqlite(self, report_path: str) -> str | None:
        for suffix in [".sqlite", ".nsys-rep.sqlite"]:
            path = report_path + suffix
            if os.path.isfile(path):
                return path
            alt = report_path.replace(".nsys-rep", "") + suffix
            if os.path.isfile(alt):
                return alt
        return None

    def _parse_sqlite(self, sqlite_path: str) -> NsysResult:
        try:
            conn = sqlite3.connect(sqlite_path)
            cursor = conn.execute(
                "SELECT duration FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY duration"
            )
            times_us = [row[0] / 1000.0 for row in cursor]
            conn.close()

            if not times_us:
                return NsysResult(error="no kernel events found")

            return NsysResult(
                median_kernel_us=times_us[len(times_us) // 2],
                all_kernel_us=times_us,
                success=True,
            )
        except Exception as e:
            return NsysResult(error=f"SQLite parse error: {e}")
