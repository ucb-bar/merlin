"""A Phase 2 gSIM cell reads outputs back as a digest only when Spike ties it to full values."""

from __future__ import annotations

import pytest
from merlin_experiments.phase2 import gsim_digest_readback as DR

from merlin.runtime.out_digest import container_bytes, xxh64

CB = {"target": "t", "tensors": {"Y": {"shape": [2, 2], "dtype": "i32", "role": "output"}}}
VALUES = {"Y": [[1, -2], [3, 4]]}


def _digest(values):
    return f"{xxh64(container_bytes([v for row in values for v in row], 4)):016x}"


class _Oracle:
    def __init__(self, tmp_path, *, gsim_values=VALUES, rebuilt_elf=False):
        self.tmp, self.gsim_values, self.rebuilt_elf, self.calls = tmp_path, gsim_values, rebuilt_elf, []

    def __call__(self, cb, llvm, *, simulator, target, workdir, timeout, readback_policy=None):
        digest = readback_policy is not None
        self.calls.append((simulator, digest))
        name = "digest" if digest else "full"
        if digest and simulator == "gsim" and self.rebuilt_elf:
            name = "rebuilt"
        elf = self.tmp / f"{simulator if not digest else 'any'}_{name}.elf"
        elf.write_bytes(name.encode())
        result = {"elf": str(elf), "cycles": 77 if simulator == "gsim" else 1, "outputs": {}}
        held = self.gsim_values if simulator == "gsim" else VALUES
        if digest:
            result["output_digests"] = {"Y": {"nbytes": 16, "digest": _digest(held["Y"]), "dtype": "i32"}}
        else:
            result["outputs"] = held
        return result


def test_matching_digests_report_the_spike_verified_values(tmp_path, monkeypatch):
    monkeypatch.delenv(DR.MODE_ENV, raising=False)
    oracle = _Oracle(tmp_path)
    res = DR.run_gsim(CB, "llvm", target="t", workdir=tmp_path / "w", timeout=5, oracle=oracle)
    assert oracle.calls == [("spike", False), ("spike", True), ("gsim", True)], "gSIM never reads values back"
    assert res["readback"]["mode"] == "digest" and res["cycles"] == 77 and res["outputs"] == VALUES
    assert res["readback"]["digests"] == {"Y": _digest(VALUES["Y"])}


@pytest.mark.parametrize(
    "oracle_kwargs, reason",
    [({"gsim_values": {"Y": [[1, -2], [3, 5]]}}, "differ"), ({"rebuilt_elf": True}, "not the ELF")],
)
def test_any_disagreement_falls_back_to_the_full_gsim_readback(tmp_path, monkeypatch, oracle_kwargs, reason):
    monkeypatch.delenv(DR.MODE_ENV, raising=False)
    oracle = _Oracle(tmp_path, **oracle_kwargs)
    res = DR.run_gsim(CB, "llvm", target="t", workdir=tmp_path / "w", timeout=5, oracle=oracle)
    assert oracle.calls[-1] == ("gsim", False), "the verdict comes from gSIM's own full values"
    assert res["readback"]["mode"] == "full" and reason in res["readback"]["reason"]
    assert res["outputs"] == oracle.gsim_values


def test_the_full_readback_can_be_kept_for_every_cell(tmp_path, monkeypatch):
    monkeypatch.setenv(DR.MODE_ENV, "full")
    oracle = _Oracle(tmp_path)
    res = DR.run_gsim(CB, "llvm", target="t", workdir=tmp_path / "w", timeout=5, oracle=oracle)
    assert oracle.calls == [("gsim", False)] and res["readback"]["mode"] == "full"


def test_capsule_grading_on_gsim_uses_digests_only_when_the_regrade_asks(tmp_path, monkeypatch):
    """The Phase 2 functional regrade sets MERLIN_GSIM_L3_READBACK=digest for its gSIM L3; without it
    (Phase 1 grading), the adapter keeps its automatic full-value readback."""
    from merlin.targetgen import oracle_readback as ORB

    calls = []

    def run(cb, llvm, *, sim, target, backend, workdir, timeout, policy):
        calls.append((sim, policy))
        return {"elf": str(tmp_path / "x.elf"), "outputs": {}}

    monkeypatch.setattr(ORB, "_run", run)
    seen = {}
    monkeypatch.setattr(DR, "run_gsim", lambda cb, llvm, **kw: seen.setdefault("digest", kw) or {"via": "digest"})
    monkeypatch.delenv(ORB.GSIM_L3_READBACK_ENV, raising=False)
    ORB.run_with_readback(CB, "llvm", sim="gsim", target="t", backend=None, workdir=tmp_path, timeout=5, policy=None)
    assert calls == [("gsim", None)] and "digest" not in seen
    monkeypatch.setenv(ORB.GSIM_L3_READBACK_ENV, "digest")
    ORB.run_with_readback(CB, "llvm", sim="gsim", target="t", backend=None, workdir=tmp_path, timeout=5, policy=None)
    assert "oracle" in seen["digest"], "the digest flow runs through the adapter's own oracle"
    ORB.run_with_readback(CB, "llvm", sim="spike", target="t", backend=None, workdir=tmp_path, timeout=5, policy=None)
    assert calls[-1] == ("spike", None), "spike is never routed through digests"
