"""Shared admission gate for Qwen isolated launch benchmark receipts."""

import json
from pathlib import Path


def admit_bench(output, expected_variants=None):
    """Require explicit timing equality and projection arithmetic qualification.

    A successful subprocess and status='ok' alone are insufficient. Other
    families may not publish an independent bit-exact golden; their timed
    outputs must still explicitly match the functional simulation.
    """
    launches = json.loads((Path(output) / "MODEL-BENCH.json").read_text())["launches"]
    failed = [row.get("key", "<missing key>") for row in launches
              if row.get("status") != "ok" or row.get("timed_output_equal") is not True
              or not row.get("family") or not row.get("key")
              or (row.get("family") == "projection_gemm" and row.get("bit_exact") is not True)]
    keys = [row.get("key") for row in launches]
    if (not launches or failed or len(set(keys)) != len(keys)
            or (expected_variants is not None and len(launches) != expected_variants)):
        raise RuntimeError(f"Qwen launch qualification failed: {len(launches)} variants; {failed}")
    return dict(status="admitted", variants=len(launches), timed_output_equal=True,
                projection_bit_exact=True)
