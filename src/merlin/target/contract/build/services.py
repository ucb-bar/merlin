"""Pure build services: no oracle registry, execution engine or reference imports.

Only trusted host adapters construct these objects. A serialized candidate
request is not a renderer, recipe, or filesystem authorization.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import shlex
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from types import FunctionType, MethodType, ModuleType

from merlin.target.contract.harness.readout import _symbol_name

# ---- build_service -----------------------------------------------------------------------------


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_build_package(package_init: Path) -> ModuleType:
    """Load a host-selected pure target package WITHOUT discovering any backend.

    Paths come from the installed target adapter, never an agent action. The
    complete package's Python file set/bytes are checked on every reuse. A
    changed host package requires a fresh process; old class identities are not
    silently mixed with new source bytes.
    """
    path = Path(package_init).absolute()
    if path.is_symlink() or path.name != "__init__.py" or path.resolve() != path or not path.is_file():
        raise ValueError("pure build package needs a real absolute initializer")
    paths = sorted(path.parent.rglob("*.py"))
    if any(item.is_symlink() or item.resolve() != item for item in paths):
        raise ValueError("pure build source must not escape through symlinks")
    pins = {str(item): file_digest(item) for item in paths}
    namespace = "merlin._pure_build_packages"
    name = namespace + ".p_" + hashlib.sha256(str(path).encode()).hexdigest()
    if name in sys.modules:
        module = sys.modules[name]
        if getattr(module, "_merlin_build_source_pins", None) != pins:
            raise ValueError("loaded pure build package changed; start a fresh host process")
        return module
    if namespace not in sys.modules:
        parent = ModuleType(namespace)
        parent.__path__ = []
        sys.modules[namespace] = parent
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[str(path.parent)])
    if spec is None or spec.loader is None:
        raise ValueError("pure build package has no Python source loader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        if any(file_digest(Path(item)) != pin for item, pin in pins.items()):
            raise ValueError("pure build package changed while loading")
        module._merlin_build_source_pins = pins
        return module
    except BaseException:
        for loaded in list(sys.modules):
            if loaded == name or loaded.startswith(name + "."):
                del sys.modules[loaded]
        raise


@dataclass(frozen=True)
class BuildOnlyService:
    """Typed host capability used only by the optional cache-free compile branch.

    The renderer consumes explicit inputs and returns C text. It does not run a
    reference or emulator. Source pins are obligations supplied by the trusted
    adapter; they do not grant paths to a sandbox or qualify emitted numerics.
    """

    target: str
    recipe: object
    renderer: Callable
    source_pins: tuple[tuple[str, str], ...]

    def verify(self, target: str) -> None:

        if (
            type(self) is not BuildOnlyService
            or self.target != target
            or type(self.recipe) is not HarnessBuildRecipe
            or not callable(self.renderer)
            or not self.source_pins
            or len(dict(self.source_pins)) != len(self.source_pins)
        ):
            raise ValueError("build-only service must be a typed target-bound host capability")
        for path, expected in self.source_pins:
            item = Path(path)
            if (
                not item.is_absolute()
                or any(parent.is_symlink() for parent in (item, *item.parents))
                or item.resolve() != item
                or not item.is_file()
                or file_digest(item) != expected
            ):
                raise ValueError("build-only source/tool pin changed: " + str(path))
        renderer = self.renderer
        while type(renderer) is partial:
            renderer = renderer.func
        if type(renderer) not in (FunctionType, MethodType):
            raise ValueError("build-only renderer must be an actual Python function or bound method")
        owner = inspect.getsourcefile(renderer)
        declared = dict(self.source_pins)
        if owner is None or not Path(owner).is_absolute() or str(Path(owner)) not in declared:
            raise ValueError("build-only renderer has no pinned inspected source owner")
        if file_digest(Path(owner)) != declared[str(Path(owner))]:
            raise ValueError("build-only renderer inspected source owner changed")

    def render(self, cb, *, target, inputs, warm_profile=None, readback_policy=None, blobs=None):
        self.verify(target)
        # An explicit empty map denotes a zero-input program. Its renderer must
        # still check the complete argument/output boundary against that map.
        if not isinstance(inputs, dict):
            raise ValueError("build-only service requires explicit logical inputs")
        kwargs = {"inputs": inputs}
        # A generic **kwargs forwarding wrapper is not proof that the PURE
        # renderer behind it accepts sidecars. Require an explicit capability.
        if blobs is not None and "blobs" in inspect.signature(self.renderer).parameters:
            kwargs["blobs"] = blobs
        if warm_profile is not None:
            kwargs["warm_profile"] = warm_profile
        if readback_policy is not None:
            from merlin.target.contract.harness.readback_policy import selected

            selected(readback_policy)
            if "readback_policy" not in inspect.signature(self.renderer).parameters:
                raise NotImplementedError("build-only renderer has no explicit full-value readback capability")
            kwargs["readback_policy"] = readback_policy
        result = self.renderer(cb, **kwargs)
        self.verify(target)
        if not isinstance(result, str):
            raise ValueError("build-only renderer did not return source text")
        return result


# ---- build_recipe ------------------------------------------------------------------------------
# Pure compile/link recipe shared by build-only and runtime services.

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


# ---- compile_only ------------------------------------------------------------------------------
# No-data linkage checks for the ordinary compiled pointer entry ABI.
#
# These declarations and products prove only structural compilation. The caller
# owns independently derived original IR/ABI provenance and semantic, index,
# resource and output-store proofs. No tensor values or goldens enter this owner.


@dataclass(frozen=True)
class CompileOnlyTensor:
    """One original tensor slot, derived by the protected source producer."""

    name: str
    shape: tuple[int, ...]
    dtype: str

    def record(self):
        if (
            type(self.name) is not str
            or not self.name
            or type(self.shape) is not tuple
            or any(type(size) is not int or size < 0 for size in self.shape)
            or type(self.dtype) is not str
            or not self.dtype
        ):
            raise ValueError("compile-only original ABI needs named static tensor types")
        return {"name": self.name, "shape": list(self.shape), "dtype": self.dtype}


@dataclass(frozen=True)
class CompileOnlySourceAbi:
    """Immutable original roster, including zero inputs; not proof authority."""

    inputs: tuple[CompileOnlyTensor, ...]
    outputs: tuple[CompileOnlyTensor, ...]

    def record(self):
        for role, slots in (("input", self.inputs), ("output", self.outputs)):
            if (
                type(slots) is not tuple
                or (role == "output" and not slots)
                or any(type(slot) is not CompileOnlyTensor for slot in slots)
            ):
                raise ValueError("compile-only source ABI needs complete immutable input/output slots")
            if len({slot.name for slot in slots}) != len(slots):
                raise ValueError("compile-only source ABI repeats an original input/output")
        return {"inputs": [slot.record() for slot in self.inputs], "outputs": [slot.record() for slot in self.outputs]}

    def bind(self, cb):
        from merlin.runtime.engine.command_buffer import whole_program_entry_bindings
        from merlin.target.contract.interfaces.grammar import match_tensor_spec

        self.record()
        tensors, kernel = cb.get("tensors"), cb.get("kernel_abi")
        if type(tensors) is not dict or type(kernel) is not dict or kernel.get("kind") != "whole_program":
            raise ValueError("compile-only candidate has no whole-program pointer ABI")
        args, outputs = kernel.get("args"), kernel.get("outputs")
        if type(args) is not list or not args or type(outputs) is not list or len(outputs) != len(set(outputs)):
            raise ValueError("compile-only candidate omits or repeats an ABI argument/output")
        slots = {}
        for argument in args:
            if (
                type(argument) is not dict
                or set(argument) != {"tensor", "access"}
                or argument["access"] not in {"read", "write", "readwrite"}
                or type(argument["tensor"]) is not str
                or argument["tensor"] in slots
            ):
                raise ValueError("compile-only candidate has an ambiguous pointer slot")
            slots[argument["tensor"]] = argument["access"]
        leaves = whole_program_entry_bindings(cb)
        if leaves is None:
            leaves = [name for name, access in slots.items() if access != "write"]
        if (
            len(leaves) != len(self.inputs)
            or len(outputs) != len(self.outputs)
            or set(leaves) != {name for name, access in slots.items() if access != "write"}
            or set(outputs) != {name for name, access in slots.items() if access != "read"}
        ):
            raise ValueError("compile-only candidate does not cover every original input/output")
        positional = cb.get("operand_naming") == "positional" or cb.get("interface") == "linalg_positional"
        if not positional:
            if set(leaves) != {slot.name for slot in self.inputs} or set(outputs) != {
                slot.name for slot in self.outputs
            }:
                raise ValueError("compile-only candidate changes named original inputs/outputs")
            leaves, outputs = [slot.name for slot in self.inputs], [slot.name for slot in self.outputs]
        bindings = []
        for role, originals, emitted in (("input", self.inputs, leaves), ("output", self.outputs, outputs)):
            for original, name in zip(originals, emitted, strict=True):
                match_tensor_spec(original.record(), tensors.get(name))
                bindings.append({"role": role, "source": original.name, "emitted": name})
        return {"original_abi": self.record(), "bindings": bindings, "pointer_arity": len(args)}


@dataclass(frozen=True)
class CompileOnlyLinkage:
    """A host-created retained reference with no operand storage or execution."""

    entry_symbol: str
    pointer_arity: int
    command_buffer_sha256: str

    def verify(self, cb):
        if (
            not _symbol_name(self.entry_symbol)
            or type(self.pointer_arity) is not int
            or self.pointer_arity <= 0
            or self.pointer_arity != len((cb.get("kernel_abi") or {}).get("args") or ())
            or self.command_buffer_sha256
            != hashlib.sha256(json.dumps(cb, sort_keys=True, allow_nan=False).encode()).hexdigest()
        ):
            raise ValueError("compile-only linkage differs from its exact emitted pointer declaration")

    def render(self, cb):
        self.verify(cb)
        parameters = ", ".join("void *" for _ in range(self.pointer_arity))
        arguments = ", ".join("(void *)0" for _ in range(self.pointer_arity))
        # An opaque volatile gate keeps the ordinary kernel and its dependencies
        # reachable through linker garbage collection. The probe is never run.
        return (
            f"extern void {self.entry_symbol}({parameters});\n"
            "static volatile unsigned char merlin_compile_only_gate;\n"
            "int main(void) {\n"
            f"  if (merlin_compile_only_gate) {self.entry_symbol}({arguments});\n"
            "  return 0;\n}\n"
        )


def require_pointer_entry(lowered_mlir, *, entry_symbol, pointer_arity):
    """Require the actual plain C pointer signature used by the shared harness.

    This checks the emitted declaration, not source equivalence, body effects,
    device placement or runtime support. Opaque operations remain unqualified;
    the ordinary stock translator still verifies its complete input artifact.
    """
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import NoneAttr
    from xdsl.utils.exceptions import ParseError, VerifyException

    from merlin.compiler.ir.llvm_context import parse_llvm_module

    if not _symbol_name(entry_symbol) or type(pointer_arity) is not int or pointer_arity < 1:
        raise ValueError("emitted entry requires an explicit symbol and positive pointer ABI arity")
    try:
        module = parse_llvm_module(lowered_mlir)
    except (ValueError, ParseError, VerifyException) as error:
        raise ValueError("emitted entry has malformed LLVM pointer ABI IR") from error
    entries = [
        operation
        for operation in module.walk()
        if isinstance(operation, llvm.FuncOp) and operation.sym_name.data == entry_symbol
    ]
    if len(entries) != 1:
        raise ValueError("emitted LLVM has no unique selected entry")
    entry = entries[0]
    if (
        not entry.body.blocks
        or entry.function_type.is_variadic
        or type(entry.function_type.output) is not llvm.LLVMVoidType
        or entry.CConv.convention.data != "ccc"
        or entry.linkage.linkage.data != "external"
        or len(entry.function_type.inputs) != pointer_arity
        or tuple(value.type for value in entry.body.blocks[0].args) != tuple(entry.function_type.inputs)
        or any(
            type(value) is not llvm.LLVMPointerType or type(value.addr_space) is not NoneAttr
            for value in entry.function_type.inputs
        )
        or any(attributes.data for attributes in entry.arg_attrs or ())
        or any(attributes.data for attributes in entry.res_attrs or ())
    ):
        raise ValueError("emitted entry does not define the selected ordinary C pointer ABI")


def prepare_linkage(*, cb, lowered_mlir, entry_symbol, original_abi):
    """Check pointer ABI structure; stock translation still verifies LLVM semantics."""
    if type(original_abi) is not CompileOnlySourceAbi:
        raise ValueError("compile-only linkage requires a typed independently derived original ABI")
    bindings = original_abi.bind(cb)
    require_pointer_entry(lowered_mlir, entry_symbol=entry_symbol, pointer_arity=bindings["pointer_arity"])
    linkage = CompileOnlyLinkage(
        entry_symbol,
        bindings["pointer_arity"],
        hashlib.sha256(json.dumps(cb, sort_keys=True, allow_nan=False).encode()).hexdigest(),
    )
    linkage.verify(cb)
    return linkage, bindings


# ---- pointer_storage ---------------------------------------------------------------------------
# Explicit original logical pointer storage, derived without tensor allocation.
#
# The policy is selected independently of compiler output. These typed values are
# software ABI declarations, not provenance, actual allocation or physical proof.
# Source authorities and emitted-code checkers must establish those separately.


def _integer_bits(dtype):
    from merlin.common import target_data as quant_formats

    bits = quant_formats.machine_bits(dtype)
    if bits is not None and any(
        dtype.startswith(prefix) and dtype[len(prefix) :].isdecimal() for prefix in ("int", "uint", "i", "u")
    ):
        return bits
    raise ValueError("original pointer storage currently needs raw integer element types")


@dataclass(frozen=True)
class PointerStoragePolicy:
    """Closed explicit software choices; missing choices have no inferred default."""

    layout: str
    argument_order: str
    slot_aliasing: str
    extent_derivation: str
    byte_order: str
    tensor_alignment: int

    def record(self):
        if (
            self.layout != "row_major_contiguous"
            or self.argument_order != "inputs_then_outputs"
            or self.slot_aliasing != "disjoint"
            or self.extent_derivation != "static_original_tensor_type"
            or self.byte_order not in {"little", "big"}
            or type(self.tensor_alignment) is not int
            or self.tensor_alignment < 1
            or self.tensor_alignment & (self.tensor_alignment - 1)
        ):
            raise ValueError("original pointer storage requires every explicit supported software ABI choice")
        return dict(self.__dict__)


@dataclass(frozen=True)
class PointerStorageSlot:
    ordinal: int
    role: str
    name: str
    shape: tuple[int, ...]
    dtype: str
    element_bits: int
    element_count: int
    byte_extent: int
    element_strides: tuple[int, ...]

    def record(self):
        return {**self.__dict__, "shape": list(self.shape), "element_strides": list(self.element_strides)}


@dataclass(frozen=True)
class OriginalPointerStorageContract:
    """Original complete logical slots under an independently selected policy.

    Each pointer must denote the complete declared object when executed. The
    contract does not prove physical storage exists, fits a machine, stays alive
    or matches an independently qualified runtime. A checker may prove static
    code obligations under these original software ABI preconditions.
    """

    original_abi: CompileOnlySourceAbi
    policy: PointerStoragePolicy

    @property
    def slots(self):
        if type(self.original_abi) is not CompileOnlySourceAbi or type(self.policy) is not PointerStoragePolicy:
            raise ValueError("original pointer storage needs exact typed original ABI and explicit policy")
        self.original_abi.record()
        self.policy.record()
        slots, names = [], set()
        for role, tensors in (("input", self.original_abi.inputs), ("output", self.original_abi.outputs)):
            for tensor in tensors:
                bits = _integer_bits(tensor.dtype)
                if (
                    bits % 8
                    or bits < 8
                    or any(extent <= 0 for extent in tensor.shape)
                    or not tensor.shape
                    or tensor.name in names
                ):
                    raise ValueError(
                        "original pointer storage currently needs distinct positive byte-sized integer slots"
                    )
                names.add(tensor.name)
                strides, stride = [], 1
                for extent in reversed(tensor.shape):
                    strides.append(stride)
                    stride *= extent
                count = math.prod(tensor.shape)
                slots.append(
                    PointerStorageSlot(
                        len(slots),
                        role,
                        tensor.name,
                        tensor.shape,
                        tensor.dtype,
                        bits,
                        count,
                        count * (bits // 8),
                        tuple(reversed(strides)),
                    )
                )
        return tuple(slots)

    def record(self):
        return {
            "schema": "merlin.original_pointer_storage.v1",
            "policy": self.policy.record(),
            "original_abi": self.original_abi.record(),
            "slots": [slot.record() for slot in self.slots],
            "scope": (
                "original software ABI preconditions; physical allocation, lifetime and runtime equivalence unproved"
            ),
        }

    def bind_candidate(self, cb):
        """Check actual call ordering against the original software declaration.

        Candidate metadata selects neither the policy nor original extents. Its
        pointer roster is checked only as the actual shared harness call binding.
        Emitted memory/index/output semantics still require an independent check.
        """
        slots = self.slots
        binding = self.original_abi.bind(cb)
        expected = [
            {"tensor": row["emitted"], "access": "read" if row["role"] == "input" else "write"}
            for row in binding["bindings"]
        ]
        if cb["kernel_abi"]["args"] != expected or len(expected) != len(slots):
            raise ValueError("candidate pointer order/access differs from the original disjoint slot ABI")
        return binding
