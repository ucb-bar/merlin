"""Per-workload wall watchdogs; measured target cycles are never timeout labels."""

import json
import math
import os
import statistics
import tempfile
import threading
from pathlib import Path


def positive_seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("Watchdog durations must be positive finite seconds")
    return float(value)


class MedianWatchdog:
    """Three baseline timings, then 3x median of the last nine correct runs.

    One campaign owns this file; its runner threads share this instance. Bind
    histories to exact program/fixture/compiler/graph/hardware identities.
    Failed and cached runs are not observations. Admitted jobs keep their limit.
    """

    def __init__(self, path, *, bootstrap_seconds, ceiling_seconds, binding):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.policy = dict(kind="median_wall_v1", baseline_samples=3, history_samples=9,
                           multiplier=3, bootstrap_seconds=positive_seconds(bootstrap_seconds),
                           ceiling_seconds=positive_seconds(ceiling_seconds))
        if self.policy["bootstrap_seconds"] > self.policy["ceiling_seconds"]:
            raise ValueError("Bootstrap watchdog exceeds the hard ceiling")
        expected = dict(policy=self.policy, binding=binding)
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if any(self.state.get(k) != v for k, v in expected.items()):
                raise ValueError("Timeout history belongs to another policy or measurement context")
            for row in self.state["workloads"].values():
                if len(row["baseline"]) > 3 or len(row["recent"]) > 9:
                    raise ValueError("Invalid timeout history lengths")
                for value in row["baseline"] + row["recent"]:
                    positive_seconds(value)
        else:
            self.state = dict(**expected, workloads={})

    def timeout(self, key, *, baseline=False):
        with self.lock:
            row = self.state["workloads"].get(key, dict(baseline=[], recent=[]))
            if baseline:
                if len(row["baseline"]) >= 3:
                    raise ValueError("Baseline watchdog already calibrated for this workload")
                return self.policy["bootstrap_seconds"]
            if len(row["baseline"]) != 3:
                raise ValueError("Need three successful baseline timings before candidate search")
            floor = 3 * statistics.median(row["baseline"])
            ceiling = self.policy["ceiling_seconds"]
            if floor > ceiling:
                raise ValueError("Hard ceiling is below 3x baseline median; review the pilot configuration")
            median = statistics.median(row["recent"]) if len(row["recent"]) == 9 else statistics.median(row["baseline"])
            return min(ceiling, max(floor, 3 * median))

    def observe(self, key, seconds, *, baseline=False):
        seconds = positive_seconds(seconds)
        with self.lock:
            row = self.state["workloads"].setdefault(key, dict(baseline=[], recent=[], successes=0))
            if baseline:
                if len(row["baseline"]) >= 3:
                    raise ValueError("Baseline watchdog already calibrated for this workload")
                row["baseline"].append(seconds)
            else:
                if len(row["baseline"]) != 3:
                    raise ValueError("Cannot add candidates before baseline calibration")
                row["recent"] = (row["recent"] + [seconds])[-9:]
            row["successes"] += 1
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent, delete=False) as stream:
                json.dump(self.state, stream, indent=2)
                stream.write("\n")
                temporary = stream.name
            os.replace(temporary, self.path)
