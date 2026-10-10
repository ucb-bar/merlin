"""Phase 1's cost plane REPORTS a movement floor beside its issue floor: the command buffer's declared
bytes over the memory-path widths of the elaboration the timing engine's own build receipt names. It
never enters the verdict."""

from __future__ import annotations

import hashlib
import json

from merlin.perf import cost_plane as CP

_CMD = (
    "{ flip ready : UInt<1>, valid : UInt<1>, bits : { inst : { funct : UInt<7>, rs2 : UInt<5>, "
    "opcode : UInt<7>}, rs1 : UInt<64>}}"
)
_EDGE = (
    "mem : { a : { flip ready : UInt<1>, valid : UInt<1>, bits : { opcode : UInt<3>, address : UInt<32>, "
    "mask : UInt<16>, data : UInt<128>}}, flip d : { flip ready : UInt<1>, valid : UInt<1>, bits : "
    "{ opcode : UInt<3>, data : UInt<128>}}}"
)


def _engine(tmp_path, *, tamper=False):
    fir = tmp_path / "model.fir"
    fir.write_text(
        "FIRRTL version 4.0.0\ncircuit Top :\n  module Accel : @[gen/x.scala 1:1]\n"
        f"    output auto : {{ {_EDGE}}} @[x.scala 1:1]\n"
        f"    output io : {{ flip cmd : {_CMD}, busy : UInt<1>}} @[x.scala 1:1]\n    wire w : UInt<1>\n"
    )
    digest = hashlib.sha256(fir.read_bytes()).hexdigest()
    (tmp_path / "build_receipt.json").write_text(
        json.dumps({"artifacts": {"firrtl": {"path": str(fir), "sha256": "0" * 64 if tamper else digest}}})
    )
    return {"sim_provenance": {"binary": str(tmp_path / "emulator")}}


def test_the_engine_receipt_yields_the_memory_path_and_a_movement_floor(tmp_path):
    memory = CP.engine_memory_path(_engine(tmp_path))
    assert memory["status"] == "derived" and memory["read_bytes_per_cycle"] == 16
    row = {"movement_volume": {"known_bytes_in": 512, "known_bytes_out": 1024}}
    fields = CP.movement_fields(row, memory, compute_floor=16.0)
    assert fields["movement_floor_cycles"] == 64.0 and fields["limiter"] == "movement"
    assert "512 in, 1024 out" in fields["movement_basis"]


def test_an_unverifiable_engine_or_an_undeclared_volume_is_unknown(tmp_path):
    assert CP.engine_memory_path(_engine(tmp_path, tamper=True))["status"] == "unknown"
    assert CP.engine_memory_path({})["status"] == "unknown"
    memory = {"status": "derived", "read_bytes_per_cycle": 16, "write_bytes_per_cycle": 16}
    assert CP.movement_fields({}, memory, compute_floor=1.0)["movement_floor_cycles"] is None
