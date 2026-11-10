"""Actual consecutive production schedules and checked complete handoffs."""

from __future__ import annotations

import json

import test_lean_graph_state as fixture


def actual_chain(root, width=8, tail=False, arrangement="cross-config"):
    first = root / "first"
    first.mkdir()
    first_order = [0, 2, 1, 3, 4, 5] if arrangement == "cross-config" else [2, 0, 1, 3, 4, 5]
    if tail:
        first_order.append(6)
    before, middle, contract, produced = fixture.actual(first, width, tail, first_order)
    second = root / "second"
    second.mkdir()
    source = second / "source.mlir"
    source.write_bytes(middle.read_bytes())
    selected = second / "contract.json"
    selected.write_bytes(contract.read_bytes())
    # Actual second production transformation consumes EXACT first output.
    command = [
        fixture.sys.executable,
        str(fixture.repo_root() / "src/merlin/verify/lean_graph_state_capture.py"),
        "--reference",
        str(fixture.REFERENCE),
        "--source",
        str(source),
        "--contract",
        str(selected),
        "--destination",
        str(second / "actual"),
        "--order",
        *map(str, [1, 0, 2, 3, 4, 5, 6] if tail else [1, 0, 2, 3, 4, 5]),
    ]
    process = fixture.subprocess.run(
        command,
        cwd=fixture.REFERENCE,
        env={**fixture.os.environ, "PYTHONPATH": str(fixture.REFERENCE / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
    )
    (second / "producer.stdout").write_text(process.stdout)
    (second / "producer.stderr").write_text(process.stderr)
    assert process.returncode == 0, process.stderr
    capture = json.loads((second / "actual/capture.json").read_text())
    assert capture["projection_checked"] and capture["producer"].endswith("planning.apply_order")
    return [(before, middle), (source, second / "actual/after.mlir")], contract, produced
