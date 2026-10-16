"""Implementation overrides cannot hide outside the namespace or behind an auxiliary name."""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest


@pytest.mark.parametrize(
    "attack",
    [
        "outside_namespace",
        "auxiliary_name",
        "assumptions_namespace",
        "unsafe_helper",
        "partial_helper",
        "extern_helper",
        "safe_helper",
    ],
)
def test_audit_checks_defining_modules(tmp_path, lean_checker, attack):
    lean = shutil.which("lean")
    if lean is None:
        if os.environ.get("MERLIN_LEAN_REQUIRED"):
            pytest.fail("Lean compiler unavailable")
        pytest.skip("Lean compiler unavailable")
    body = {
        "outside_namespace": """
namespace Escape
def accept (_ : Bool) : Bool := true
@[implemented_by accept] def preserve (b : Bool) : Bool := b
end Escape
namespace MerlinVerify
def probe (b : Bool) : Bool := Escape.preserve b
theorem probe_sound (b : Bool) (h : probe b = true) : b = true := h
end MerlinVerify
""",
        "auxiliary_name": """
namespace MerlinVerify
def base (b : Bool) : Bool := b
def accept (_ : Bool) : Bool := true
@[implemented_by accept] def base._unsafe_rec (b : Bool) : Bool := b
end MerlinVerify
""",
        "safe_helper": """
namespace Escape
def preserve (b : Bool) : Bool := b
end Escape
namespace MerlinVerify
def probe (b : Bool) : Bool := Escape.preserve b
theorem probe_sound (b : Bool) (h : probe b = true) : b = true := h
end MerlinVerify
""",
        "assumptions_namespace": """
namespace MerlinVerify.Assumptions
axiom unreviewed : False
end MerlinVerify.Assumptions
theorem MerlinVerify.bogus : False := MerlinVerify.Assumptions.unreviewed
""",
        "unsafe_helper": "unsafe def Escape.unchecked : Bool := true\n",
        "partial_helper": "partial def Escape.loop (n : Nat) : Nat := Escape.loop n\n",
        "extern_helper": '@[extern "unreviewed_external"] opaque Escape.foreign : UInt32\n',
    }[attack]
    source = tmp_path / "MerlinVerify" / "GuardProbe.lean"
    source.parent.mkdir()
    source.write_text("import Lean\n" + body)
    subprocess.run(
        [lean, "-R", str(tmp_path), "-o", str(source.with_suffix(".olean")), str(source)],
        check=True,
        capture_output=True,
        text=True,
    )
    env = dict(os.environ, LEAN_PATH=str(tmp_path))
    proc = subprocess.run(
        [str(lean_checker.with_name("merlin-lean-audit")), "MerlinVerify.GuardProbe"],
        env=env,
        capture_output=True,
        text=True,
    )
    report = json.loads(proc.stdout)
    if attack == "safe_helper":
        assert proc.returncode == 0 and not report["violations"]
        assert report["theorems_audited"] == 1
    else:
        assert proc.returncode == 1
        expected = {
            "assumptions_namespace": "axiom outside a reviewed Assumptions module:",
            "unsafe_helper": "unsafe:",
            "partial_helper": "partial:",
            "extern_helper": "extern:",
        }.get(attack, "implemented_by:")
        assert any(expected in v for v in report["violations"])
