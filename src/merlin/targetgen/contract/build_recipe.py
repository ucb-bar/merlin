"""Pure compile/link recipe shared by build-only and runtime services."""

from __future__ import annotations

import shlex
import subprocess
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

_RISCV_ABIS = frozenset({"ilp32", "ilp32e", "ilp32f", "ilp32d", "lp64", "lp64f", "lp64d"})


def named_object_paths(sources: Sequence[Path], workdir: Path) -> tuple[Path, ...]:
    """Assign stable distinct objects without overwriting caller-supplied objects.

    Unique source basenames retain their historical output names. Collisions
    use the original link-position ordinal; link order and recipe are unchanged.
    This purely names outputs and performs no compilation or backend discovery.
    """
    sources = tuple(map(Path, sources))
    workdir = Path(workdir)
    suffixes = {".c", ".S", ".s"}
    counts = Counter(source.stem for source in sources if source.suffix in suffixes)
    occupied = {source.resolve() for source in sources if source.suffix not in suffixes}
    preserved = {
        stem for stem, count in counts.items() if count == 1 and (workdir / f"{stem}.o").resolve() not in occupied
    }
    occupied.update((workdir / f"{stem}.o").resolve() for stem in preserved)
    outputs = []
    for index, source in enumerate(sources):
        if source.suffix not in suffixes:
            outputs.append(source)
            continue
        if source.stem in preserved:
            outputs.append(workdir / f"{source.stem}.o")
            continue
        candidate = workdir / f"{index}_{source.stem}.o"
        retry = 0
        while candidate.resolve() in occupied:
            retry += 1
            candidate = workdir / f"{index}_{source.stem}_{retry}.o"
        occupied.add(candidate.resolve())
        outputs.append(candidate)
    return tuple(outputs)


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
    # Opt-in headers compiled into the harness.  Each is a direct child of an
    # include root; the ELF cache hashes its exact bytes rather than treating
    # the include-root path alone as a source identity.
    header_dependencies: tuple[Path, ...] = ()
    # Support sources (a subset of ``support_sources``) whose code must be linked AHEAD of the harness
    # and the candidate's objects: a startup routine that reaches them with a short-range branch (a
    # RISC-V ``j``/``jal`` spans +-1 MiB) breaks once candidate code between them grows past that.
    link_first: tuple[Path, ...] = ()

    def ordered_link_sources(self, sources: Sequence[Path]) -> list[Path]:
        """``link_first`` support sources, then ``sources``, then the remaining support sources."""
        first = [Path(s) for s in self.link_first]
        if any(f not in {Path(s) for s in self.support_sources} for f in first):
            raise self.error_cls("link_first names a source that is not one of the recipe's support sources")
        rest = [Path(s) for s in self.support_sources if Path(s) not in first]
        return [*first, *(Path(s) for s in sources), *rest]

    def require_kernel_stack_frame(self) -> KernelStackFramePolicy:
        """Return the target-declared policy or refuse to compile a target-bound kernel."""
        if type(self.kernel_stack_frame) is not KernelStackFramePolicy:
            raise self.error_cls(
                "this target's build recipe declares no kernel stack-frame policy; the runner "
                "cannot prove that the package entrypoint fits the target runtime stack"
            )
        return self.kernel_stack_frame

    def command(self, *, sources: Sequence[Path], output: Path, link_script: Path | None = None) -> list[str]:
        """The full compiler invocation for ``sources`` -> ``output``."""
        cmd = [str(self.compiler), *self.cflags]
        for root in self.include_roots:
            cmd += ["-I", str(root)]
        cmd += ["-T", str(link_script or self.link_script), "-o", str(output)]
        cmd += [str(s) for s in self.ordered_link_sources(sources)]
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
            "kernel for the same ISA the harness is built for, and a mismatch is a runtime trap"
        )

    def mabi(self) -> str:
        """Resolve the effective RISC-V ABI from recipe flags or the selected compiler.

        A missing explicit flag is not an ABI default chosen by the runner. It is a fact of this
        compiler with these flags, queried without compiling a source. The result is then made
        explicit for both halves of the ELF by :meth:`with_effective_abi`.
        """
        declared: list[str] = []
        flags = iter(self.cflags)
        for flag in flags:
            if flag.startswith("-mabi="):
                declared.append(flag.partition("=")[2])
            elif flag == "-mabi":
                declared.append(next(flags, ""))
        if len(declared) > 1:
            raise self.error_cls("build recipe has ambiguous duplicate -mabi options")
        if declared:
            abi = declared[0]
        else:

            def diagnostic(command: list[str], answer: subprocess.CompletedProcess[str]) -> str:
                # Preserve the actual selected query, not just the last parser's
                # refusal. A failed GCC query can precede a successful driver
                # trace that has no Clang cc1 ABI; losing the first diagnostic
                # makes that infrastructure failure impossible to distinguish.
                return (
                    f"exit={answer.returncode}, stderr={answer.stderr[-600:]!r}, "
                    f"stdout={answer.stdout[-600:]!r}, command={shlex.join(command)!r}"
                )

            command = [str(self.compiler), *self.cflags, "-Q", "--help=target"]
            try:
                answer = subprocess.run(command, capture_output=True, text=True, timeout=10)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise self.error_cls(
                    f"cannot query selected compiler's effective ABI: command={shlex.join(command)!r}: {exc}"
                ) from exc
            first_diagnostic = diagnostic(command, answer)
            if answer.returncode == 0:
                values = [
                    parts[1]
                    for line in answer.stdout.splitlines()
                    if (parts := line.split()) and len(parts) == 2 and parts[0] == "-mabi="
                ]
                if len(values) != 1:
                    raise self.error_cls("selected compiler returned no unique effective -mabi: " + first_diagnostic)
                abi = values[0]
            else:
                # Clang reports its resolved cc1 target ABI in a dry-run driver trace.
                command = [str(self.compiler), *self.cflags, "-###", "-x", "c", "-c", "/dev/null", "-o", "/dev/null"]
                try:
                    answer = subprocess.run(command, capture_output=True, text=True, timeout=10)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    raise self.error_cls(
                        f"cannot query selected compiler's effective ABI: {first_diagnostic}; "
                        f"driver trace command={shlex.join(command)!r}: {exc}"
                    ) from exc
                query_diagnostics = first_diagnostic + "; driver trace " + diagnostic(command, answer)
                if answer.returncode:
                    raise self.error_cls("selected compiler cannot report its effective -mabi: " + query_diagnostics)
                values = []
                for line in answer.stderr.splitlines():
                    args = shlex.split(line)
                    if "-cc1" not in args:
                        continue
                    values += [args[index + 1] for index, arg in enumerate(args[:-1]) if arg == "-target-abi"]
                if len(values) != 1:
                    raise self.error_cls("selected compiler returned no unique effective -mabi: " + query_diagnostics)
                abi = values[0]
        march = self.march().partition("=")[2]
        if (
            abi not in _RISCV_ABIS
            or not march.startswith(("rv32", "rv64"))
            or not abi.startswith("ilp32" if march.startswith("rv32") else "lp64")
        ):
            raise self.error_cls(f"build recipe has invalid -mabi={abi!s} for -march={march}")
        return f"-mabi={abi}"

    def with_effective_abi(self) -> HarnessBuildRecipe:
        """Pin the selected ABI in each harness compile/link command, including implicit defaults."""
        abi = self.mabi()
        flags: list[str] = []
        original = iter(self.cflags)
        for flag in original:
            if flag == "-mabi":
                next(original, None)
            elif not flag.startswith("-mabi="):
                flags.append(flag)
        return replace(self, cflags=(*flags, abi))

    def link_command(self, *, objects: Sequence[Path], output: Path, link_script: Path | None = None) -> list[str]:
        """Link already-compiled objects. Support sources are NOT re-appended: they are among them."""
        cmd = [str(self.compiler), *self.cflags]
        for root in self.include_roots:
            cmd += ["-I", str(root)]
        cmd += ["-T", str(link_script or self.link_script), "-o", str(output)]
        return cmd + [str(o) for o in objects] + list(self.ldflags)
