"""WHICH Spike extension a target's L2 functional oracle loads — read from that target's own contract,
verified by digest, fail-closed.

THE FAILURE THIS EXISTS TO PREVENT. A Spike RoCC extension is compiled against a *specific* generated
``gemmini_params.h``. Two sibling elaborations of one generator differ in that header (mesh dimension,
accumulator geometry, whether the full-width accumulator read port exists at all), and a functional
model built from one header running a kernel compiled against the other does not fail loudly: it
returns wrong numbers. The repo has already paid for this once — a params.h skew produced an
all-zeros matmul that read as a compiler defect (memory ``libgemmini-driver-allzeros-header-skew``).

So the extension is not a property of "whichever chipyard the operator's ``$MERLIN_CHIPYARD`` names".
It is a property of THE TARGET, and a target that has its own model declares it in its own contract::

    runner:
      spike_extension:
        extension_name: <the extension class the .so registers>
        extlib_env: <operator-provided absolute artifact path variable>
        sha256: <64 hex>

STRICTLY ADDITIVE, AND THAT IS LOAD-BEARING. A target that declares no ``runner.spike_extension`` is
resolved exactly as before: the caller's own default library directory and default extension name, with
no ``--extlib`` flag. Nothing about such a target's invocation changes, byte for byte
(``tests/gemmini/test_spike_extension.py`` asserts that against the live reference backend).

FAIL CLOSED, NEVER SUBSTITUTE. When a target DOES declare one, every way of not getting exactly those
bytes raises :class:`SpikeExtensionError`:

* the declaration is incomplete (neither ``extlib`` nor ``extlib_env``, no
  ``extension_name`` or ``sha256``);
* the file is absent or unreadable;
* the bytes on disk do not hash to the declared ``sha256``.

The one thing this must never do is fall back to another target's ``.so``. A target may
declare an ``extlib_env`` location so the operator can place the artifact outside any
particular user's checkout, but the SHA-256 remains target-owned: the environment can
relocate identical bytes, never authorize different model bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---- spike_extension ---------------------------------------------------------------------------


class SpikeExtensionError(RuntimeError):
    """A target DECLARED a Spike extension and the declared bytes could not be produced.

    Deliberately not a subclass of any target's backend error: the condition is about the ORACLE's
    identity, not about a program, and a caller that swallows backend errors must not swallow this.
    """


@dataclass(frozen=True)
class SpikeExtension:
    """The resolved L2 functional-model identity for one target.

    ``declared`` distinguishes the two cases a reader must never confuse: a target whose own contract
    names and digests its model (``True``), and a target that has no declaration and is therefore
    served by whatever the caller's toolchain default resolves to (``False``).
    """

    target: str
    extension_name: str
    #: The ``.so`` to pass as ``--extlib``, or ``None`` when the caller's default search path is used.
    extlib: Path | None
    #: The directory that must be on ``LD_LIBRARY_PATH`` for the extension to load.
    library_dir: Path
    #: The digest the contract DECLARES, and which the bytes on disk were verified against.
    sha256: str | None
    declared: bool

    def spike_flags(self) -> tuple[str, ...]:
        """The ``spike`` argv flags, in the order the extension API requires (``--extlib`` before
        ``--extension``; spike parses the library before it can look the extension name up)."""
        if self.extlib is None:
            return (f"--extension={self.extension_name}",)
        return (f"--extlib={self.extlib}", f"--extension={self.extension_name}")

    def describe(self) -> str:
        if not self.declared:
            return (
                f"{self.target}: toolchain-default spike extension {self.extension_name!r} from "
                f"{self.library_dir} (target declares none)"
            )
        return (
            f"{self.target}: declared spike extension {self.extension_name!r} at {self.extlib} (sha256 {self.sha256})"
        )


def _contract_runner(target: str) -> dict[str, Any]:
    """``runner`` from ``target``'s contract, or ``{}`` when the contract cannot be read.

    An UNREADABLE contract is not the same condition as an INVALID declaration and must not raise:
    a caller with no contract at all (a bare backend probe, an out-of-tree package that has not been
    generated) has always resolved to its own default, and turning that into an exception would break
    every such caller for a fact none of them declares. An unreadable contract yields "declares none",
    and the fail-closed half below then never runs.
    """
    try:
        from merlin.target.registry import identity as target_registry

        contract = target_registry.resolve(target).load_contract()
    except Exception:  # noqa: BLE001 — no contract / unresolvable target ⇒ "declares none", see above
        return {}
    if not isinstance(contract, dict):
        return {}
    runner = contract.get("runner")
    return runner if isinstance(runner, dict) else {}


def declared_extension(target: str) -> dict[str, Any] | None:
    """The raw ``runner.spike_extension`` block ``target`` declares, or ``None``.

    ``None`` means "this target names no model of its own" — the additive default path. A block that is
    PRESENT but malformed is returned as-is so :func:`resolve` can refuse it with a precise reason
    rather than silently reading it as an absence.
    """
    block = _contract_runner(target).get("spike_extension")
    if block is None:
        return None
    if not isinstance(block, dict):
        raise SpikeExtensionError(
            f"{target}: runner.spike_extension must be a mapping with extension_name/extlib/sha256, "
            f"got {type(block).__name__}"
        )
    return block


#: Verified (path, size, mtime_ns) triples. The KEY carries the stat fields, so a rebuilt or replaced
#: ``.so`` is re-hashed rather than trusted: the cache saves the repeated hash inside one grade, and
#: cannot carry a verification across a change to the bytes. (Deliberately not keyed on the path alone.)
_verified_stat: dict[tuple[str, int, int], str] = {}


def _digest_verified(target: str, so: Path, declared_sha: str) -> str:
    try:
        st = so.stat()
    except OSError as exc:
        raise SpikeExtensionError(
            f"{target}: declared spike extension {so} cannot be stat'ed ({exc}). The L2 functional "
            f"model this target's verdicts are measured on is absent; refusing to run on another "
            f"target's extension"
        ) from exc
    key = (str(so), st.st_size, st.st_mtime_ns)
    cached = _verified_stat.get(key)
    if cached is not None:
        got = cached
    else:
        h = hashlib.sha256()
        try:
            with so.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(chunk)
        except OSError as exc:
            raise SpikeExtensionError(f"{target}: declared spike extension {so} cannot be read ({exc})") from exc
        got = h.hexdigest()
        _verified_stat[key] = got
    if got != declared_sha:
        raise SpikeExtensionError(
            f"{target}: declared spike extension {so} has sha256 {got}, but the contract declares "
            f"{declared_sha}. These are DIFFERENT bytes, so this is a different functional model than "
            f"the one this target's contract was reviewed against — refusing to grade on it. Either the "
            f"file was rebuilt (update runner.spike_extension.sha256 with a reviewed rationale) or the "
            f"path now names somebody else's build"
        )
    return got


def resolve(target: str, *, default_library_dir: str | Path, default_extension_name: str) -> SpikeExtension:
    """The verified L2 extension identity for ``target``.

    ``default_library_dir`` / ``default_extension_name`` are the CALLER's status quo, used unchanged
    when the target declares nothing — the parameters exist so this module names no toolchain layout and
    no extension name of its own.

    Raises :class:`SpikeExtensionError` when the target declares an extension that cannot be produced
    exactly as declared.
    """
    block = declared_extension(target)
    if block is None:
        return SpikeExtension(
            target=target,
            extension_name=str(default_extension_name),
            extlib=None,
            library_dir=Path(default_library_dir),
            sha256=None,
            declared=False,
        )
    name = block.get("extension_name")
    extlib = block.get("extlib")
    extlib_env = block.get("extlib_env")
    sha = block.get("sha256")
    missing = [
        field
        for field, value in (("extension_name", name), ("sha256", sha))
        if not isinstance(value, str) or not value.strip()
    ]
    if not extlib and not extlib_env:
        missing.append("extlib")
    if missing:
        raise SpikeExtensionError(
            f"{target}: runner.spike_extension is declared but incomplete — missing {missing}. A model "
            f"a verdict cites must be named AND digested; an undigested path cannot be told apart from "
            f"a sibling elaboration's build sitting in the same directory"
        )
    if extlib and extlib_env:
        raise SpikeExtensionError(f"{target}: runner.spike_extension cannot declare both extlib and extlib_env")
    if extlib_env:
        if not isinstance(extlib_env, str) or not extlib_env.isidentifier():
            raise SpikeExtensionError(f"{target}: runner.spike_extension.extlib_env must name one environment variable")
        selected = os.environ.get(extlib_env, "").strip()
        if not selected:
            raise SpikeExtensionError(
                f"{target}: runner.spike_extension requires {extlib_env} with the absolute path to its "
                "SHA-pinned .so; refusing to fall back to the default Spike extension"
            )
        extlib = selected
    so = Path(str(extlib).strip()).expanduser()
    if not so.is_absolute():
        raise SpikeExtensionError(
            f"{target}: runner.spike_extension.extlib must be an absolute path, got {extlib!r} — a "
            f"relative one resolves against whatever directory the grade happens to run from"
        )
    declared_sha = str(sha).strip().lower()
    if len(declared_sha) != 64 or any(c not in "0123456789abcdef" for c in declared_sha):
        raise SpikeExtensionError(f"{target}: runner.spike_extension.sha256 must be 64 hex characters, got {sha!r}")
    if not so.is_file():
        raise SpikeExtensionError(
            f"{target}: declared spike extension {so} is not a file. This target's L2 oracle is its own "
            f"model; refusing to fall back to the toolchain default at {default_library_dir}, which is "
            f"a DIFFERENT elaboration's extension and would put another device's verdict on this "
            f"target's submission"
        )
    _digest_verified(target, so, declared_sha)
    return SpikeExtension(
        target=target,
        extension_name=str(name).strip(),
        extlib=so,
        library_dir=so.parent,
        sha256=declared_sha,
        declared=True,
    )


def spike_invocation(
    target: str, *, default_library_dir: str | Path, default_extension_name: str
) -> tuple[tuple[str, ...], Path]:
    """``(spike argv flags, the directory to put on LD_LIBRARY_PATH)`` for ``target``.

    The one call a backend's ``run_elf`` needs. For a target that declares nothing this returns exactly
    ``((f"--extension={default_extension_name}",), Path(default_library_dir))`` — the status quo.
    """
    ext = resolve(target, default_library_dir=default_library_dir, default_extension_name=default_extension_name)
    return ext.spike_flags(), ext.library_dir


def library_path_with(library_dir: str | Path, env: dict[str, str] | None = None) -> str:
    """``library_dir`` prepended to ``LD_LIBRARY_PATH`` — the same string shape callers already build,
    factored out so the extension directory and the search path cannot drift apart."""
    existing = (env or os.environ).get("LD_LIBRARY_PATH", "")
    return f"{library_dir}:{existing}"


# ---- runtime_build -----------------------------------------------------------------------------
# Self-contained, RTL-derived runtime build support for the bare-metal L2/L3 oracle.
#
# The runtime splits cleanly into two kinds of configuration:
#
# * **DERIVED hardware facts** — the platform (SoC) DRAM base, i.e. the load address the bare-metal linker
#   must use. This is read from the TARGET'S OWN RTL BUILD memory map, never baked: a new HW RTL repo gets
#   its runtime layout for free. (:func:`platform_dram_base`.)
# * **Operator/setup config** — where the RTL build, toolchain, and sim binaries live. That is the person
#   setting up the board's choice, so it stays in the target descriptor / ``.env`` (``sim_via``, the
#   chipyard location via ``ext_path``, the sim config via the capability manifest), NOT derived here.
#
# Dispatch is by the RTL build tool (``sim_via``, a descriptor fact), mirroring how the oracle adapters are
# chosen — so this holds no target-name literal and extends to another build tool by adding a reader.


def rtl_sim_config(target: str) -> str | None:
    """The declared chipyard harness config for ``target`` (capability manifest ``runtime.rtl_sim_config``)
    — a per-target FACT read from the registry, not a hardcoded constant. It names the elaborated design
    every chipyard simulator of the target is built from (``simulator-``/``simv-chipyard.harness-<cfg>``).
    None when the target declares none or its manifest cannot be read."""
    try:
        from merlin.target.registry.experiment_descriptor import load_capability_manifest

        return (load_capability_manifest(target).contract.get("runtime") or {}).get("rtl_sim_config")
    except Exception:  # noqa: BLE001 — manifest unavailable ⇒ no config; caller falls back
        return None


def _chipyard_dram_base(target: str) -> int | None:
    """Derive the platform DRAM base from the target's chipyard RTL build memory map: the base of the
    largest ``memory@`` region in the generated ``<config>.memmap.json``. The chipyard location is a setup
    fact (``MERLIN_CHIPYARD`` / ``.env`` / ``ext_path``); the config is a manifest fact. Returns None if the
    build/memmap is absent (the caller uses a documented fallback), never a baked address."""
    cfg = rtl_sim_config(target)
    if not cfg:
        return None
    from merlin.common.paths import env as _env
    from merlin.common.paths import ext_path as _ext_path

    cy = _env("MERLIN_CHIPYARD") or _ext_path("chipyard")
    if not cy:
        return None
    hw = f"chipyard.harness.TestHarness.{cfg}"
    mm = Path(cy) / "sims" / "verilator" / "generated-src" / hw / f"{hw}.memmap.json"
    if not mm.is_file():
        return None
    try:
        regions = json.loads(mm.read_text()).get("mapping", [])
    except Exception:  # noqa: BLE001 — malformed memmap ⇒ no derivation, fall back
        return None
    mems = [r for r in regions if (r.get("names") or [""])[0].startswith("memory@")]
    if not mems:
        return None
    biggest = max(mems, key=lambda r: (r.get("size") or [0])[0])
    base = (biggest.get("base") or [None])[0]
    return int(base) if base is not None else None


# The bare-metal DRAM base used when the RTL memory map cannot be read (build absent). It is the RISC-V
# platform reset/DRAM base every Rocket/Chipyard-class SoC and spike/fesvr use — a documented default, not
# a per-target guess; the derived value from the RTL build always wins when available.
DEFAULT_PLATFORM_DRAM_BASE = 0x80000000  # derived-ok: RISC-V platform DRAM base; platform_dram_base() prefers the RTL-derived value  # fmt: skip


def platform_dram_base(target: str, sim_via: str | None) -> int:
    """The platform (SoC) DRAM base for ``target`` — the load address the bare-metal linker uses. DERIVED
    from the RTL build's memory map, dispatched by the RTL build tool (``sim_via``): chipyard reads its
    ``memmap.json`` ``memory@`` region. Falls back to :data:`DEFAULT_PLATFORM_DRAM_BASE` only when the
    build/memmap is unavailable. Keyed on the sim ENGINE's ``has_memmap`` capability, not its NAME. No
    per-target address is baked here."""
    from merlin.target.simulators.oracle_policy import sim_oracle_caps

    caps = sim_oracle_caps(sim_via)
    derived = _chipyard_dram_base(target) if (caps is not None and caps.has_memmap) else None
    return derived if derived is not None else DEFAULT_PLATFORM_DRAM_BASE


def compiler_smoke(sim_via: str | None) -> tuple[bool, str]:
    """Pre-spend check that the RTL-oracle COMPILE toolchain actually WORKS — not merely that its binaries
    exist. It compiles a trivial LLVM-IR module to a riscv object with the oracle's own clang and, when
    the simulator declares an MLIR Python dependency, imports its required bindings in that interpreter.
    Missing or broken tools become NO_GO before a paid run tool-crashes on every capsule.
    Only for a compile-based sim (its ``_SimOracle.is_compile_based`` capability); other oracles return
    n/a. Keyed on the capability, not the engine NAME."""
    from merlin.target.simulators.oracle_policy import sim_oracle_caps

    caps = sim_oracle_caps(sim_via)
    if caps is None or not caps.is_compile_based:
        return True, "n/a (no compile-based oracle for this sim)"
    try:
        from merlin.compiler.lower.model import toolchain as _tc

        clang = _tc.clang()
    except Exception as e:  # noqa: BLE001
        return False, f"clang toolchain unresolved: {e}"
    cp = Path(str(clang))
    if cp.is_absolute() and not cp.exists():
        return False, f"oracle clang not found at {clang} — the compile toolchain is not provisioned"
    with tempfile.TemporaryDirectory() as td:
        ll = Path(td) / "smoke.ll"
        obj = Path(td) / "smoke.o"
        ll.write_text("define i32 @f() {\nentry:\n  ret i32 0\n}\n", encoding="utf-8")
        try:
            r = subprocess.run(
                [str(clang), "--target=riscv64-unknown-elf", "-march=rv64gc", "-c", str(ll), "-o", str(obj)],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except FileNotFoundError:
            return False, f"oracle clang missing/not executable: {clang}"
        except Exception as e:  # noqa: BLE001
            return False, f"compile smoke could not run: {str(e)[-160:]}"
        if r.returncode != 0 or not obj.is_file():
            return False, f"oracle clang failed to compile a riscv object: {(r.stderr or '')[-200:]}"
    if caps.requires_mlir_python:
        python = _tc.compiler_python()
        if not python.is_file():
            return False, (
                f"oracle MLIR Python not found at {python}; select MERLIN_COMPILER_PYTHON "
                "or MERLIN_COMPILER_VENV before launching"
            )
        try:
            result = subprocess.run(
                [str(python), "-I", "-c", "from torch_mlir import ir, passmanager"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"oracle MLIR Python could not start: {exc}"
        if result.returncode != 0:
            return False, f"oracle MLIR Python cannot import torch-mlir bindings: {(result.stderr or '')[-200:]}"
    detail = f"oracle clang compiles riscv objects ({cp.name})"
    return True, f"{detail}; MLIR Python ready" if caps.requires_mlir_python else detail


def _rebase_ld(text: str, base: int) -> str | None:
    """Replace the FIRST absolute location-counter assignment ``. = 0x...;`` in a linker script with the
    derived ``base`` — structurally (str ops, no regex). Returns None if no such origin is found (the
    caller then uses the template unchanged, never a wrong rewrite)."""
    key = ". = 0x"
    i = text.find(key)
    if i < 0:
        return None
    j = text.find(";", i)
    if j < 0:
        return None
    return text[:i] + f". = {hex(base)};" + text[j + 1 :]


def derived_link_script(base: int, template_ld: Path, out_dir: Path) -> Path:
    """Emit a bare-metal linker script whose load address is the DERIVED platform DRAM ``base`` (from
    :func:`platform_dram_base`), reusing the proven section layout of ``template_ld`` (the target's
    curated-harness linker script — a per-target setup fact the crt expects) but replacing its baked origin
    with the derived value. Layout stays exactly what the runtime needs; only the base becomes derived, not
    baked. If the template has no rewritable origin, it is copied through unchanged (fail-safe)."""
    text = template_ld.read_text(encoding="utf-8")
    rebased = _rebase_ld(text, base)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "link.derived.ld"
    out.write_text(rebased if rebased is not None else text, encoding="utf-8")
    return out


__all__ = [
    "SpikeExtension",
    "SpikeExtensionError",
    "declared_extension",
    "resolve",
    "spike_invocation",
]
