"""The build half of the contract-compile seam: a target declares HOW to build, core only orchestrates.

The generic contract-compile path did not merely emit one target's harness text — it ran that target's
entire build. Compiler, include roots, march, link script, support-source globs and even the exception
type raised on failure all came from importing one backend, so a target whose toolchain differed could
not use the path at all, no matter what its contract said.

The recipe is an OPTIONAL backend capability. A backend that never builds a bare-metal harness simply
does not define it, and a caller that needs one is told which target lacks it rather than getting an
AttributeError from somewhere deeper.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from merlin.runtime.backends import base


def _recipe(**over):
    kw = dict(compiler=Path("/opt/cc"), include_roots=(Path("/inc/a"), Path("/inc/b")),
              support_sources=(Path("/sup/x.c"), Path("/sup/y.S")), link_script=Path("/ld/test.ld"),
              load_address=0x8000_0000, cflags=("-O2", "-static"), error_cls=ValueError)
    kw.update(over)
    return base.HarnessBuildRecipe(**kw)


def test_the_command_places_every_declared_piece():
    cmd = _recipe().command(sources=[Path("/w/harness.c"), Path("/w/kernel.o")],
                            output=Path("/w/out.elf"))
    assert cmd[0] == "/opt/cc", "the compiler must lead the invocation"
    for flag, arg in (("-I", "/inc/a"), ("-I", "/inc/b"), ("-T", "/ld/test.ld"), ("-o", "/w/out.elf")):
        assert any(cmd[i] == flag and cmd[i + 1] == arg for i in range(len(cmd) - 1)), (flag, arg)
    assert cmd.index("/w/harness.c") < cmd.index("/sup/x.c"), "support sources come after the inputs"
    for piece in ("-O2", "-static", "/w/kernel.o", "/sup/y.S"):
        assert piece in cmd, piece


def test_an_overriding_link_script_wins():
    """The load address is derived from the RTL memory map at build time, so the caller supplies a
    rewritten script rather than the declared one — the declared one only names the section layout."""
    cmd = _recipe().command(sources=[Path("/w/h.c")], output=Path("/w/o.elf"),
                            link_script=Path("/w/link.derived.ld"))
    assert "/w/link.derived.ld" in cmd and "/ld/test.ld" not in cmd


def test_the_error_class_travels_with_the_recipe():
    """A build failure should raise what that target's callers already catch, not a generic error
    they would have to start handling."""
    assert _recipe().error_cls is ValueError
    assert base.HarnessBuildRecipe(
        compiler=Path("/cc"), include_roots=(), support_sources=(), link_script=Path("/l"),
        load_address=0).error_cls is RuntimeError


# ------------------------------------------------------------------ the registry lookup
def test_a_backend_without_a_recipe_refuses_by_name():
    """A host/simulator backend legitimately has none; the refusal must say which target and why."""
    with pytest.raises(NotImplementedError, match="spike"):
        base.harness_build_recipe("spike")


def test_the_reference_target_declares_a_complete_recipe():
    """The regression for the migration: if the backend stops declaring one, the generic path loses
    its build entirely rather than falling back to the literals it used to carry."""
    r = base.harness_build_recipe("gemmini")
    assert r.compiler.name and r.include_roots and r.support_sources
    assert r.load_address > 0 and r.cflags
    assert issubclass(r.error_cls, Exception) and r.error_cls is not RuntimeError
