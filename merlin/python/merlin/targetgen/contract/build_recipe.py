"""Pure compile/link recipe shared by build-only and runtime services."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KernelStackFramePolicy:
    """A target-owned static stack limit for one package kernel entrypoint.

    Neither the entry symbol nor the usable stack reservation can be inferred from LLVM IR or from
    accelerator RTL.  They are software-ABI facts, so the target that supplies the bare-metal build
    recipe supplies both.  Keeping them together also lets a pure build service carry the policy
    without importing an execution backend or rediscovering a target contract.
    """

    entry_symbol: str
    max_static_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.entry_symbol, str) or not self.entry_symbol:
            raise ValueError("kernel stack-frame policy requires a nonempty entry symbol")
        # ``bool`` is an ``int`` in Python, but accepting it here would turn True into a one-byte
        # safety policy and produce a very misleading compile refusal.
        if type(self.max_static_bytes) is not int or self.max_static_bytes <= 0:
            raise ValueError("kernel stack-frame policy requires a positive integer byte budget")

    def record(self) -> dict[str, object]:
        """Stable serialization used by pure-build capabilities and build-cache identities."""
        return {"entry_symbol": self.entry_symbol, "max_static_bytes": self.max_static_bytes}


@dataclass(frozen=True)
class HarnessBuildRecipe:
    """How to compile + link a runner-owned harness against one target's bare-metal environment.

    The generic contract-compile path used to obtain every one of these by importing a specific
    backend, which meant it did not merely emit one target's harness text — it ran one target's entire
    build. None of it is derivable from RTL: a compiler path, an include layout and a set of support
    sources are properties of a target's software environment, so the backend that owns the target
    supplies them and the generic path only orchestrates.

    ``error_cls`` travels with the recipe so a build failure still raises the exception type that
    target's callers already catch, rather than a generic one they would have to start handling.
    """

    compiler: Path
    include_roots: tuple[Path, ...]
    support_sources: tuple[Path, ...]
    link_script: Path
    load_address: int
    cflags: tuple[str, ...] = ()
    error_cls: type[Exception] = RuntimeError
    # Ordered trailing linker arguments, including library/archive groups. Keeping these
    # separate avoids losing static-library symbols by scanning archives before their users.
    ldflags: tuple[str, ...] = ()
    # Optional at construction so host-only/legacy recipe users retain their API.  A target-bound
    # LLVM object build requires it and fails closed when it is absent; ``target=None`` is the legacy
    # unbound object-only path and deliberately has no authority to invent a target's stack limit.
    kernel_stack_frame: KernelStackFramePolicy | None = None

    def require_kernel_stack_frame(self) -> KernelStackFramePolicy:
        """Return the target-declared policy or refuse to compile a target-bound kernel."""
        if type(self.kernel_stack_frame) is not KernelStackFramePolicy:
            raise self.error_cls(
                "this target's build recipe declares no kernel stack-frame policy; the runner "
                "cannot prove that the package entrypoint fits the target runtime stack")
        return self.kernel_stack_frame

    def command(self, *, sources: "Sequence[Path]", output: Path,
                link_script: Path | None = None) -> list[str]:
        """The full compiler invocation for ``sources`` -> ``output``."""
        cmd = [str(self.compiler), *self.cflags]
        for root in self.include_roots:
            cmd += ["-I", str(root)]
        cmd += ["-T", str(link_script or self.link_script), "-o", str(output)]
        cmd += [str(s) for s in sources]
        cmd += [str(s) for s in self.support_sources]
        cmd += self.ldflags
        return cmd

    def compile_command(self, *, source: Path, output: Path) -> list[str]:
        """Compile ONE source to an object with an explicit name.

        Compiling and linking in a single invocation makes the build non-reproducible: the driver
        names its intermediate object ``ccXXXXXX.o`` and that random name is recorded in the ELF as an
        STT_FILE symbol. Measured 2026-09-03: two builds of byte-identical sources differed in exactly
        6 bytes, ``ccFzUU8w.o`` vs ``ccnuEDwa.o``, while producing identical cycle counts. That single
        difference defeats any content-addressed reuse of a measurement, because the artifact digest
        moves when nothing about the program did.
        """
        cmd = [str(self.compiler), *self.cflags]
        for root in self.include_roots:
            cmd += ["-I", str(root)]
        return cmd + ["-c", str(source), "-o", str(output)]

    def march(self) -> str:
        """The ISA string this target's bare-metal build declares (``-march=...``), or a refusal.

        Read rather than assumed, because it has to agree with the OTHER half of the same ELF. The
        runner compiles the package's kernel object itself, and that step carried its own hardcoded
        march: on a core whose recipe says ``rv64gc`` the kernel was built ``rv64gcv``, so the moment
        a kernel had anything the auto-vectorizer could take (a scalar host-lane float program is the
        first one that does), it emitted ``vsetivli``/``vle32.v`` for a core with no vector unit and
        trapped -- reported as the submission's kernel faulting at runtime.
        """
        for flag in self.cflags:
            if flag.startswith("-march="):
                return flag
        raise self.error_cls(
            "this target's build recipe declares no -march=; the runner cannot compile the package "
            "kernel for the same ISA the harness is built for, and a mismatch is a runtime trap")

    def link_command(self, *, objects: "Sequence[Path]", output: Path,
                     link_script: Path | None = None) -> list[str]:
        """Link already-compiled objects. Support sources are NOT re-appended: they are among them."""
        cmd = [str(self.compiler), *self.cflags]
        for root in self.include_roots:
            cmd += ["-I", str(root)]
        cmd += ["-T", str(link_script or self.link_script), "-o", str(output)]
        return cmd + [str(o) for o in objects] + list(self.ldflags)
