"""Which Spike extension a target's L2 oracle loads — target-parameterized, digest-verified, fail-closed.

Two properties, and the FIRST one is the reason this file exists at the same time as the second.

1. THE STATUS QUO IS UNCHANGED FOR A TARGET THAT DECLARES NOTHING. The live reference backend resolves
   `--extension=<its own name>` with `chipyard_root()/.conda-env/riscv-tools/lib` on the library path,
   exactly as it did before `runner.spike_extension` existed. Asserted against the REAL backend and the
   REAL contract, not against a fixture, because the value of the assertion is that it would notice a
   change to either.

2. A TARGET THAT DECLARES ONE GETS THOSE BYTES OR AN EXCEPTION. Never a fallback to a sibling
   elaboration's `.so`: two elaborations of one generator differ in the generated `gemmini_params.h`,
   and a model built from the wrong one returns wrong numbers rather than failing (the documented
   all-zeros params-skew). So absence, a digest mismatch, an incomplete declaration and a relative
   path each raise.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from merlin.common.paths import merlin_dir
from merlin.targetgen import spike_extension as SX

# ---------------------------------------------------------------------------------------------------
# 1. the declaring target: gemmini_universal names its own model, with a digest
# ---------------------------------------------------------------------------------------------------
UNIVERSAL_CONTRACT = merlin_dir() / "targets/gemmini_universal/contracts/target_contract.yaml"


def _declared_block() -> dict:
    doc = yaml.safe_load(UNIVERSAL_CONTRACT.read_text(encoding="utf-8"))
    return (doc.get("runner") or {}).get("spike_extension") or {}


def test_the_declaring_target_declares_a_complete_digested_extension():
    """The contract carries name + absolute path + a 64-hex digest. An undigested path could not be
    told apart from a sibling elaboration's build sitting in the same directory."""
    block = _declared_block()
    assert block, f"{UNIVERSAL_CONTRACT} declares no runner.spike_extension"
    assert block.get("extension_name")
    assert Path(str(block["extlib"])).is_absolute()
    sha = str(block["sha256"]).lower()
    assert len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)


def test_declared_extension_is_read_from_the_contract_not_from_the_environment():
    """`declared_extension` reads the target's own contract. The point is that no env var can
    re-point the functional model a verdict was earned on."""
    block = SX.declared_extension("gemmini_universal")
    assert block is not None
    assert block == _declared_block()


@pytest.mark.skipif(
    not Path(str(_declared_block().get("extlib", "/nonexistent"))).is_file(),
    reason="the declared extension .so is not present on this host",
)
def test_the_declared_extension_resolves_to_its_own_verified_bytes():
    """Resolution yields --extlib BEFORE --extension (spike must load the library before it can look
    the extension name up), the library dir is the .so's OWN parent, and the bytes hash to the
    declared digest."""
    ext = SX.resolve("gemmini_universal", default_library_dir="/does/not/matter", default_extension_name="not-this-one")
    assert ext.declared is True
    assert ext.extlib is not None and ext.extlib.is_file()
    assert ext.library_dir == ext.extlib.parent
    assert ext.extension_name == _declared_block()["extension_name"]
    flags = ext.spike_flags()
    assert flags == (f"--extlib={ext.extlib}", f"--extension={ext.extension_name}")
    on_disk = hashlib.sha256(ext.extlib.read_bytes()).hexdigest()
    assert on_disk == ext.sha256 == str(_declared_block()["sha256"]).lower()
    # And the caller's default was NOT used anywhere.
    assert "not-this-one" not in " ".join(flags)
    assert "/does/not/matter" != str(ext.library_dir)


# ---------------------------------------------------------------------------------------------------
# 2. the non-declaring target: byte-identical to the status quo
# ---------------------------------------------------------------------------------------------------
def _reference_backend():
    from merlin.runtime.backends import base as backends

    try:
        return backends.get_backend("gemmini")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the reference backend is not resolvable here: {exc}")


def test_the_reference_target_declares_no_extension():
    """The guard on the whole additive claim: the moment this target DOES declare one, the byte-identity
    assertion below stops being about the fallback path and must be revisited deliberately."""
    assert SX.declared_extension("gemmini") is None


def test_the_reference_targets_spike_resolution_is_byte_identical_to_the_status_quo():
    """THE LIVE-RUN SAFETY ASSERTION. The reference backend's spike invocation must be exactly what it
    was before the resolver existed: one `--extension=<name>` flag and `libgemmini_dir()` as the
    library directory — the two values `run_elf` used to build inline."""
    gem = _reference_backend()
    flags, libdir = gem.spike_extension()
    assert flags == (f"--extension={gem.SPIKE_EXTENSION_NAME}",)
    assert libdir == gem.libgemmini_dir()
    # Spelled out independently of the backend's own constants, so a rename cannot make this vacuous:
    assert flags == ("--extension=gemmini",)
    assert libdir == gem.chipyard_root() / ".conda-env/riscv-tools/lib"


def test_an_unknown_target_falls_back_rather_than_raising():
    """A target with no readable contract has always resolved to the caller's own default. Turning that
    into an exception would break every caller for a fact none of them declares."""
    ext = SX.resolve("no-such-target-exists", default_library_dir="/opt/lib", default_extension_name="whatever")
    assert ext.declared is False
    assert ext.extlib is None
    assert ext.spike_flags() == ("--extension=whatever",)
    assert ext.library_dir == Path("/opt/lib")


# ---------------------------------------------------------------------------------------------------
# 3. fail closed — every way of not getting the declared bytes raises
# ---------------------------------------------------------------------------------------------------
def _resolve_with(monkeypatch, block):
    monkeypatch.setattr(SX, "declared_extension", lambda target: block)
    return lambda: SX.resolve("t", default_library_dir="/fallback/lib", default_extension_name="fallback-ext")


def test_a_missing_declared_so_raises_and_never_falls_back(monkeypatch, tmp_path):
    call = _resolve_with(
        monkeypatch, {"extension_name": "x", "extlib": str(tmp_path / "absent.so"), "sha256": "0" * 64}
    )
    with pytest.raises(SX.SpikeExtensionError) as e:
        call()
    assert "is not a file" in str(e.value)
    assert "/fallback/lib" in str(e.value)  # the refusal NAMES what it refused to fall back to


def test_a_digest_mismatch_raises(monkeypatch, tmp_path):
    so = tmp_path / "lib.so"
    so.write_bytes(b"these are not the declared bytes")
    call = _resolve_with(monkeypatch, {"extension_name": "x", "extlib": str(so), "sha256": "a" * 64})
    with pytest.raises(SX.SpikeExtensionError) as e:
        call()
    assert hashlib.sha256(so.read_bytes()).hexdigest() in str(e.value)


def test_matching_bytes_are_accepted(monkeypatch, tmp_path):
    so = tmp_path / "lib.so"
    so.write_bytes(b"exactly these bytes")
    good = hashlib.sha256(so.read_bytes()).hexdigest()
    call = _resolve_with(monkeypatch, {"extension_name": "x", "extlib": str(so), "sha256": good})
    ext = call()
    assert ext.declared and ext.sha256 == good and ext.library_dir == tmp_path


def test_an_edited_so_is_re_hashed_rather_than_carried_by_the_cache(monkeypatch, tmp_path):
    """The verification cache is keyed on (path, size, mtime), so replacing the file invalidates it.
    A cache keyed on the path alone would carry a verification across a change to the bytes."""
    so = tmp_path / "lib.so"
    so.write_bytes(b"exactly these bytes")
    good = hashlib.sha256(so.read_bytes()).hexdigest()
    call = _resolve_with(monkeypatch, {"extension_name": "x", "extlib": str(so), "sha256": good})
    assert call().sha256 == good
    so.write_bytes(b"exactly these bytes, plus a change")  # same path, different bytes and size
    with pytest.raises(SX.SpikeExtensionError):
        call()


@pytest.mark.parametrize(
    "block,reason",
    [
        ({"extlib": "/abs/lib.so", "sha256": "0" * 64}, "['extension_name']"),
        ({"extension_name": "x", "sha256": "0" * 64}, "['extlib']"),
        ({"extension_name": "x", "extlib": "/abs/lib.so"}, "['sha256']"),
        ({"extension_name": "x", "extlib": "/abs/lib.so", "sha256": "short"}, "64 hex characters"),
        ({"extension_name": "x", "extlib": "relative/lib.so", "sha256": "0" * 64}, "must be an absolute path"),
    ],
)
def test_an_incomplete_or_malformed_declaration_raises(monkeypatch, block, reason):
    """The REASON is asserted, not just the exception class. Every one of these declarations would
    also blow up later on some incidental check (a `None` path is not a file either), and a refusal
    that names the wrong cause sends the next reader to the wrong place."""
    with pytest.raises(SX.SpikeExtensionError) as e:
        _resolve_with(monkeypatch, block)()
    assert reason in str(e.value)


def test_a_non_mapping_declaration_raises(monkeypatch):
    monkeypatch.setattr(SX, "_contract_runner", lambda target: {"spike_extension": "libfoo.so"})
    with pytest.raises(SX.SpikeExtensionError):
        SX.declared_extension("t")


# ---------------------------------------------------------------------------------------------------
# 4. the resolution actually reaches the command line
# ---------------------------------------------------------------------------------------------------
def test_run_elf_builds_the_same_spike_argv_it_always_did(monkeypatch, tmp_path):
    """THE ASSERTION THAT COVERS THE LIVE RUN. A resolver that returns the right strings and a
    ``run_elf`` that ignores them are indistinguishable from the outside, so intercept the subprocess
    and read the argv and the environment the backend actually hands spike.

    The expectation is spelled as the pre-change line's own output: ``[spike, --extension=gemmini,
    elf]`` with ``libgemmini_dir()`` prepended to LD_LIBRARY_PATH.
    """
    gem = _reference_backend()
    seen = {}

    class _Proc:
        returncode = 0
        stdout = "DONE\n"
        stderr = ""

    def fake_run(cmd, **kw):
        seen["cmd"] = list(cmd)
        seen["env"] = dict(kw.get("env") or {})
        return _Proc()

    monkeypatch.setattr(gem.subprocess, "run", fake_run)
    elf = tmp_path / "kernel.elf"
    elf.write_bytes(b"\x7fELF")
    gem.run_elf(elf, simulator="spike", timeout=5)

    assert seen["cmd"] == [str(gem.spike_path()), "--extension=gemmini", str(elf)]
    assert seen["env"]["LD_LIBRARY_PATH"].startswith(str(gem.libgemmini_dir()) + ":")
