"""The harness ABI a target's contract declares: how a runner-owned harness calls into its kernel.

The contract splits responsibility — the package emits a kernel FUNCTION, the runner owns the harness
that embeds the leaf tensors, calls that function, and prints the OUT/METRIC/DONE protocol. To write
that harness the runner needs four facts it cannot derive from RTL, because they are software
conventions of the target's own test environment rather than properties of its hardware:

* the **entry symbol** the kernel is emitted under,
* the **fence symbol** (if any) that must be called before reading results back,
* the **includes** the harness needs for that environment's helpers, and
* the **cycle-window metric** name the runner's parser attributes cycles to.

Before this block existed these were literals in the generic compile path, which meant a target whose
test harness spelled any of them differently could not use that path at all. Note that they are
genuinely undeducible: nothing in a fact bundle says a kernel is called ``foo_kernel`` rather than
``bar_entry``. That is exactly why they belong in a contract the target authors, and why this module
refuses rather than guessing — a substituted default here produces a harness that compiles and then
fails to link, with the error pointing at the linker instead of at the missing declaration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from merlin.common.paths import repo_root
from merlin.compiler.lower.model import toolchain as host_toolchain

# ---- harness_abi -------------------------------------------------------------------------------


class HarnessAbiError(ValueError):
    """A target's contract does not declare a usable harness ABI (fail closed, never a default)."""


@dataclass(frozen=True)
class HarnessAbi:
    """What a runner needs in order to write a harness that calls this target's kernel."""

    #: The symbol the package's lowered MLIR defines the kernel under.
    entry_symbol: str
    #: Called after the kernel and before results are read; None when the target needs no fence.
    fence_symbol: str | None = None
    #: Header paths the harness must include, verbatim, in declaration order.
    includes: tuple[str, ...] = ()
    #: Metric name marking that the measured cycle window covers the accelerator region. None when the
    #: target does not distinguish, in which case no such METRIC line is printed — an absent metric and
    #: one printed as 0 are read identically by the parser, so emitting a falsy one says nothing.
    cycle_window_metric: str | None = None
    #: Extra declarations the harness emits before ``main`` (e.g. an ``extern`` for the entry symbol).
    extern_decls: tuple[str, ...] = field(default_factory=tuple)

    def declarations(self, *, entry_symbol: str | None = None, extern_decls: tuple[str, ...] | None = None) -> str:
        """The include + extern preamble, optionally for a verified alternate entry."""
        lines = [f'#include "{h}"' if not h.startswith("<") else f"#include {h}" for h in self.includes]
        symbol = entry_symbol or self.entry_symbol
        declarations = extern_decls
        if declarations is None:
            declarations = self.extern_decls if entry_symbol is None else ()
        lines.extend(declarations or (f"extern void {symbol}();",))
        return "\n".join(lines)

    def call(self, args: str) -> str:
        """The kernel call, plus the fence when the target declares one."""
        call = f"  {self.entry_symbol}({args});"
        return f"{call}\n  {self.fence_symbol}();" if self.fence_symbol else call

    def warm_profile_invocation(self, args: str, *, entry_symbol: str | None = None):
        """Return the target hooks required by a warm cycle profile.

        Unlike :meth:`call`, a performance window cannot silently accept a
        missing completion operation: doing so can read the end cycle while
        asynchronous target work is still outstanding.  The target contract is
        the authority for both symbols; the generic profile renderer contains
        neither one.
        """
        if not self.fence_symbol:
            raise HarnessAbiError(
                "a warm cycle profile requires the target harness ABI to declare an explicit completion/fence symbol"
            )
        from merlin.analysis.perf.measurement.warm_profile_harness import TargetInvocationHooks

        symbol = entry_symbol or self.entry_symbol
        if not isinstance(symbol, str) or not symbol:
            raise HarnessAbiError("a warm cycle profile requires a nonempty entry symbol")
        return TargetInvocationHooks(
            invoke=f"{symbol}({args});",
            complete=f"{self.fence_symbol}();",
        )

    def cycle_window_line(self) -> str:
        """The METRIC line attributing the cycle window, or empty when the target declares none."""
        if not self.cycle_window_metric:
            return ""
        return f'  printf("METRIC {self.cycle_window_metric} 1\\n");'


def from_contract(contract: dict[str, Any], *, target: str) -> HarnessAbi:
    """Read the ``harness_abi`` block, refusing anything incomplete.

    Raises rather than filling in a default. The literals this replaces were one target's, so a default
    would silently reintroduce exactly the weld the block exists to remove — and would do it in the one
    place hardest to notice, since the resulting harness looks correct until it fails to link.
    """
    block = (contract or {}).get("harness_abi")
    if not isinstance(block, dict) or not block:
        raise HarnessAbiError(
            f"target {target!r} declares no `harness_abi` block in its target contract; the runner "
            f"cannot write a harness without the entry symbol it should call. Add the block (see "
            f"src/merlin/resources/schemas/target_contract.schema.yaml) rather than relying on a default."
        )
    entry = block.get("entry_symbol")
    if not isinstance(entry, str) or not entry:
        raise HarnessAbiError(
            f"target {target!r}: harness_abi.entry_symbol is required and must name "
            f"the symbol the package's lowered MLIR defines the kernel under"
        )
    fence = block.get("fence_symbol") or None
    metric = block.get("cycle_window_metric") or None
    for name, value in (("fence_symbol", fence), ("cycle_window_metric", metric)):
        if value is not None and not isinstance(value, str):
            raise HarnessAbiError(f"target {target!r}: harness_abi.{name} must be a string or absent")
    return HarnessAbi(
        entry_symbol=entry,
        fence_symbol=fence,
        includes=tuple(block.get("includes") or ()),
        cycle_window_metric=metric,
        extern_decls=tuple(block.get("extern_decls") or ()),
    )


def lowered_plan_abi(contract: dict[str, Any], *, target: str) -> dict:
    """Public structural entry/completion description from the selected caller contract."""
    abi = from_contract(contract, target=target)
    instruction = (contract.get("logical_harness") or {}).get("completion_asm")
    if instruction is None:
        # This is structural admission, not a proof of accelerator completion.
        # A declared RISC-V toolchain permits its architectural memory fence;
        # never infer an accelerator-specific completion instruction or symbol.
        runner = contract.get("runner") or {}
        selected = runner.get(runner.get("backend"), {})
        flags = (selected.get("toolchain") or {}).get("cflags", ())
        if any(isinstance(flag, str) and flag.startswith(("-march=rv32", "-march=rv64")) for flag in flags):
            instruction = "fence"
    return {"entry_symbol": abi.entry_symbol, "completion_symbol": abi.fence_symbol, "completion_asm": instruction}


def for_target(target: str) -> HarnessAbi:
    """Resolve a target's harness ABI through the registry (the contract is the single source)."""
    from merlin.target.registry.identity import resolve

    return from_contract(resolve(target).load_contract(), target=target)


#: The kernel ABI a runner-owned caller uses unless the target contract selects another: the logical
#: ABI (``logical_kernel_abi`` in the OOT backend contract), dense logical tensors in one pointer order.
LOGICAL_KERNEL_ABI_VERSION = 2
#: The version-1 resident ABI (``src/merlin/resources/contracts/legacy/kernel_abi_v1.yaml``). Only a support package
#: whose own harness implements it selects it, with ``harness_abi.kernel_abi_version: 1``.
LEGACY_KERNEL_ABI_VERSION = 1


def kernel_abi_version(contract: dict[str, Any] | None, *, target: str) -> int:
    """The kernel ABI version ``contract`` selects: ``harness_abi.kernel_abi_version``, else the logical one.

    Absent means the logical ABI; anything other than 1 or 2 is refused rather than read as either."""
    block = (contract or {}).get("harness_abi")
    declared = block.get("kernel_abi_version") if isinstance(block, dict) else None
    if declared is None:
        return LOGICAL_KERNEL_ABI_VERSION
    if type(declared) is not int or declared not in (LEGACY_KERNEL_ABI_VERSION, LOGICAL_KERNEL_ABI_VERSION):
        raise HarnessAbiError(f"target {target!r}: harness_abi.kernel_abi_version must be 1 or 2, got {declared!r}")
    return declared


def kernel_abi_version_for(target: str) -> int:
    """The selected support's kernel ABI version; the logical ABI when no contract is selected for it."""
    from merlin.target.registry.identity import load_contract

    try:
        contract = load_contract(target)
    except Exception:  # noqa: BLE001 -- no selected contract declares a legacy ABI, so the default holds
        return LOGICAL_KERNEL_ABI_VERSION
    return kernel_abi_version(contract, target=target)


# ---- schemas -----------------------------------------------------------------------------------
# Fail-closed JSON-Schema validation against the ``src/merlin/resources/contracts/schemas/`` bundle.
#
# A single place to load + validate the contract schemas so the runner, the tests, and the
# packages all enforce the same rules. Validation raises :class:`ContractViolation` with a concise
# message; nothing here ever silently accepts a malformed artifact.


class ContractViolation(ValueError):
    """A contract artifact failed schema validation (fail-closed)."""


def contract_dir(override: str | Path | None = None) -> Path:
    """Resolve the contract dir: explicit override > $MERLIN_CONTRACT_DIR > in-repo src/merlin/resources/contracts
    (or, in an installed wheel with no checkout, the bundled ``_data/contract``)."""
    if override:
        return Path(override)
    from merlin.common.paths import contract_dir as _selected

    return _selected()


def load_schema(name: str, *, contract: str | Path | None = None) -> dict[str, Any]:
    """Load a schema by short name (``command_buffer`` -> command_buffer.schema.json)."""
    path = contract_dir(contract) / "schemas" / f"{name}.schema.json"
    if not path.is_file():
        raise FileNotFoundError(f"contract schema not found: {path}")
    schema = json.loads(path.read_text(encoding="utf-8"))
    if name == "manifest" and "common" in schema.get("$defs", {}):
        # Preserve the historical flat schema inspection API as well as v0.1
        # validation. The file owns one common definition shared with v0.2.
        common = schema["$defs"]["common"]
        common["properties"]["commands"]["required"] = schema["allOf"][1]["properties"]["commands"]["required"]
        return {key: value for key, value in schema.items() if key not in ("$defs", "allOf")} | common
    if name == "manifest_v2":
        # The published schema inherits the v0.1 common definition. Resolve that
        # one declared local reference here, without network schema discovery.
        reference = {"$ref": "manifest.schema.json#/$defs/common"}
        parents = schema.get("allOf", [])
        if parents.count(reference) != 1:
            raise ContractViolation("manifest_v2 must inherit the shared manifest definition")
        inherited = json.loads((path.parent / "manifest.schema.json").read_text(encoding="utf-8"))
        parents[parents.index(reference)] = inherited["$defs"]["common"]
    return schema


def validate(obj: Any, name: str, *, contract: str | Path | None = None) -> None:
    """Validate ``obj`` against the named schema; raise ContractViolation on any error."""
    import jsonschema

    schema = load_schema(name, contract=contract)
    # The contract schemas carry a RELATIVE "$id" (e.g. "merlin/bench_contract/foo.schema.json")
    # used only as a label. Under jsonschema's RefResolver that relative id becomes the base URI,
    # so an in-document fragment ref ("#/$defs/operand") resolves to a bogus URL and the resolver
    # tries to FETCH it ("unknown url type: ...merlin/bench_contract/..."), crashing validation
    # whenever an instruction carries rs1/rs2. Every "$ref" in these schemas is an in-document
    # fragment, so dropping "$id" (in-memory only) resolves refs against an empty base. This
    # ENABLES the operand-level checks that previously crashed; it does not weaken validation.
    schema.pop("$id", None)
    try:
        jsonschema.validate(obj, schema)
    except jsonschema.ValidationError as e:
        loc = "/".join(str(p) for p in e.absolute_path) or "<root>"
        raise ContractViolation(f"{name} schema violation at {loc}: {e.message}") from e


def _declared_dtypes(cap: Any):
    """Yield ``(location, token)`` for every element type a capsule declares."""
    for i, t in enumerate((cap.get("inputs") if isinstance(cap, dict) else None) or []):
        if isinstance(t, dict) and "dtype" in t:
            yield f"inputs/{i}/dtype", t["dtype"]
    pol = cap.get("numeric_policy") if isinstance(cap, dict) else None
    if isinstance(pol, dict) and "dtype" in pol:
        yield "numeric_policy/dtype", pol["dtype"]
    if isinstance(pol, dict) and pol.get("schema") == "merlin.numeric_policy.per_output.v1":
        for name, leaf in pol["outputs"].items():
            yield f"numeric_policy/outputs/{name}/dtype", leaf["dtype"]


def _check_output_policy_roster(cap: dict) -> None:
    """Join the explicit versioned policy to every declared output, without defaults."""
    policy = cap["numeric_policy"]
    if policy.get("schema") != "merlin.numeric_policy.per_output.v1":
        return
    attrs = cap["operation"].get("attributes", {})
    names = attrs.get("outs")
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
        or set(names) != set(policy["outputs"])
        or ("out" in attrs and attrs["out"] != names[0])
    ):
        raise ContractViolation(
            "per-output numeric policy requires the complete unique operation.attributes.outs "
            "roster, exactly one policy per output, and a consistent first out"
        )


def _tier_availability_conflicts(cap: Any, schema: dict[str, Any]) -> list[str]:
    """Availability words a capsule declares that CONTRADICT the tiers it requires.

    The schema's availability fields (``firesim``, ``vcs``, ``verilator``) each carry an
    ``x-oracle-tier`` annotation naming the tier the word governs.  Reading the pairing from the
    schema keeps the tier names and the field names out of this function entirely: adding a target
    whose deepest oracle has a different name is a schema edit, not a code edit.

    ``unavailable`` beside that tier in ``required_oracle_tiers`` is a capsule contradicting itself
    -- it demands a mandatory verdict from an oracle it has already said cannot run.  The runner
    would resolve that to ``incomplete`` forever, silently; better to refuse the capsule at load and
    make the author choose which half they meant.

    A field with no annotation yields no check: there is no tier it can be shown to contradict, and
    inventing one would be worse than the absent check.
    """
    if not isinstance(cap, dict):
        return []
    required = cap.get("required_oracle_tiers")
    if not isinstance(required, list):
        return []
    declared = {str(tier) for tier in required}
    problems: list[str] = []
    for field, spec in (schema.get("properties") or {}).items():
        if not isinstance(spec, dict):
            continue
        tier = spec.get("x-oracle-tier")
        if not isinstance(tier, str) or tier not in declared:
            continue
        if cap.get(field) == "unavailable":
            problems.append(
                f"{field}: unavailable, while required_oracle_tiers demands {tier} -- a capsule "
                f"cannot require a mandatory verdict from an oracle it declares cannot run"
            )
    return problems


def validate_capsule(cap: Any, *, contract: str | Path | None = None) -> None:
    """Validate a capsule: the JSON schema, then its dtype tokens against the numeric-format registry.

    The dtype vocabulary is deliberately NOT a closed enum in the schema. It was one, and that enum was
    both too small and partly wrong: it rejected every MX format the corpus builder emits (``mxfp8`` /
    ``mxfp6`` / ``mxfp4``) and spelled half precision ``f16``, which is a spelling the builder never
    writes (``corpus_spec._DTYPE`` canonicalizes both ``f16`` and ``fp16`` to ``fp16``). So radiance's and
    mx_gemmini's corpora could not be LOADED at all — every capsule in them failed validation — while the
    generator that wrote them saw nothing wrong, because the two vocabularies had nothing tying them
    together. Copying a format list into a shared schema is the overfit: which numeric formats exist is a
    property of ``src/merlin/resources/schemas/quant_formats.registry.yaml``, the declared single source of truth where
    a new format is a data entry. Accumulator widths beside them are plain machine types. Both are checked
    here, so an unrecognized token still fails closed with the ContractViolation the enum used to raise.
    """
    validate(cap, "capsule", contract=contract)
    _check_output_policy_roster(cap)
    conflicts = _tier_availability_conflicts(cap, load_schema("capsule", contract=contract))
    if conflicts:
        raise ContractViolation("capsule contract violation: " + "; ".join(conflicts))
    from merlin.common import target_data as qf

    for loc, tok in _declared_dtypes(cap):
        if not isinstance(tok, str) or not qf.is_element_dtype(tok):
            raise ContractViolation(
                f"capsule schema violation at {loc}: {tok!r} is not a registered numeric format "
                f"(src/merlin/resources/schemas/quant_formats.registry.yaml declares {qf.names()}) "
                f"nor a machine width (i8/i32/f32/…)"
            )


def validate_command_buffer(cb: Any, *, contract: str | Path | None = None) -> None:
    """Validate a command buffer against the schema AND the structural rules the schema cannot state.

    JSON Schema types an operand slot as a bare string, so a value naming nothing -- a shape, a type, a
    dimension list -- validates. A submission that emitted such operands with no ``tensors`` declared was
    told its command buffer was VALID and was then rejected downstream for a constraint the contract never
    expressed; it spent its session guessing spellings. A validator the submitter is told to run must fail
    on what the runner will refuse, or it is not the contract."""
    validate(cb, "command_buffer", contract=contract)
    for name, tensor in (cb.get("tensors") or {}).items():
        if "storage_encoding" not in tensor:
            continue
        from merlin.analysis.programs.storage_encoding import GroupedAxesStorage

        try:
            encoding = GroupedAxesStorage.from_dict(tensor["storage_encoding"])
            if list(encoding.physical_shape) != tensor["shape"] or encoding.dtype != tensor["dtype"]:
                raise ValueError("physical tensor shape or element type differs from its storage map")
        except (TypeError, ValueError) as exc:
            raise ContractViolation(f"command_buffer storage encoding for {name}: {exc}") from exc
    from merlin.runtime.engine.command_buffer import validate_command_buffer as _structural

    problems = [
        p
        for p in _structural(cb)
        if "declares no 'tensors'" in p or "whole-program kernel_abi" in p or "acc_scale_by_column" in p
    ]
    if problems:
        raise ContractViolation("command_buffer contract violation: " + "; ".join(problems))


def validate_manifest(man: Any, *, contract: str | Path | None = None) -> None:
    version = man.get("abi_version", "0.1") if isinstance(man, dict) else "0.1"
    if version == "0.1":
        validate(man, "manifest", contract=contract)
    elif version == "0.2":
        validate(man, "manifest_v2", contract=contract)
    else:
        raise ContractViolation(f"unsupported experiment ABI version {version!r}")


# --- {target}-parameterized generic contracts --------------------------------------------------------
# The two shared ABI contracts (mlir_oot_backend_contract.yaml, oracle_runner_contract.yaml) carry NO
# target literal — they template the target-specific tokens with the ``{target}`` placeholder (the same
# convention their ``--convert-iface-to-{target}`` argv already use). These readers resolve the active
# target at load, mirroring ``generate_prompt`` (``tool_stem = f"{target}-opt"`` / ``kernel_symbol =
# f"{target}_kernel"``): a fixed contract resolves to two targets' values by the same rule, with nothing
# baked in for any one accelerator.


def _resolve_target(obj: Any, target: str) -> Any:
    """Recursively fill the ``{target}`` placeholder token in a loaded contract (structured, no regex)."""
    if isinstance(obj, str):
        return obj.replace("{target}", target)
    if isinstance(obj, list):
        return [_resolve_target(v, target) for v in obj]
    if isinstance(obj, dict):
        return {k: _resolve_target(v, target) for k, v in obj.items()}
    return obj


#: The version-1 resident kernel ABI, kept apart from the candidate-facing OOT backend contract.
LEGACY_KERNEL_ABI = ("legacy", "kernel_abi_v1.yaml")


def legacy_kernel_abi_path(*, contract: str | Path | None = None) -> Path:
    """Where the version-1 kernel ABI (``kernel_abi``) lives under the selected contract dir."""
    return contract_dir(contract).joinpath(*LEGACY_KERNEL_ABI)


def render_legacy_kernel_abi(target: str, *, contract: str | Path | None = None) -> dict[str, Any]:
    """The version-1 ``kernel_abi`` block resolved for ``target`` (``symbol`` -> ``f"{target}_kernel"``).

    Read by Merlin's own device shim and by version-1 harness certification only; a runner-owned
    harness follows ``logical_kernel_abi`` in the OOT backend contract."""
    import yaml

    document = yaml.safe_load(legacy_kernel_abi_path(contract=contract).read_text(encoding="utf-8"))
    block = document.get("kernel_abi") if isinstance(document, dict) else None
    return _resolve_target(block if isinstance(block, dict) else {}, target)


def render_oracle_runner_contract(target: str, *, contract: str | Path | None = None) -> dict[str, Any]:
    """The oracle-runner contract resolved for ``target`` — the oracle-ladder level names
    (``spike_{target}_functional`` / ``{target}_verilator_rtl``) and the ``{target}_region`` cycle window
    fill from the active target. The first target resolves byte-identically to the former hand-authored names."""
    import yaml

    text = (contract_dir(contract) / "oracle_runner_contract.yaml").read_text(encoding="utf-8")
    return _resolve_target(yaml.safe_load(text), target)


def render_contract_text(name: str, target: str, *, contract: str | Path | None = None) -> str:
    """The raw contract text (``name`` is the yaml basename) with every ``{target}`` placeholder filled —
    including ones that live in YAML comments (e.g. the ``merlin.runtime.backends.{target}.parse_output``
    reference), for the agent-facing rendering the structured readers above cannot reach."""
    text = (contract_dir(contract) / name).read_text(encoding="utf-8")
    return text.replace("{target}", target)


# ---- toolchain ---------------------------------------------------------------------------------
# MLIR toolchain resolution for the experiment ABI (env-overridable).
#
# Phase 0 builds a standalone LLVM/MLIR 23 install (from ``third_party/llvm-project``) into
# ``third_party/llvm-install``. Out-of-tree C++ packages (``<target>-opt``) link against it via
# ``find_package(MLIR REQUIRED CONFIG)`` with ``MLIR_DIR`` pointing at its cmake export; the runner
# uses ``mlir-translate`` from it to take a package's lowered LLVM-dialect MLIR to ``.ll``.
#
# All paths are overridable by ``MERLIN_MLIR_INSTALL``. The matching ``clang-23`` (same LLVM 23) is
# reused from :mod:`merlin.compiler.lower.model.toolchain`.

# repo root = .../merlin (this file: merlin/python/merlin/development/targetgen/contract/toolchain.py)
_REPO = repo_root()
DEFAULT_MLIR_INSTALL = _REPO / "third_party" / "llvm-install"

# LLVM/MLIR source pin (third_party/llvm-project); recorded into run manifests + the contract.
LLVM_VERSION = "23.0.0git"
LLVM_COMMIT = "a47bddccec30"


def mlir_install() -> Path:
    """The same selected LLVM/MLIR prefix as host and device object compilation."""
    return host_toolchain.llvm_install()


def mlir_bin(tool: str) -> Path:
    """Path to a tool under the install's bin/ (mlir-opt, mlir-tblgen, mlir-translate)."""
    return mlir_install() / "bin" / tool


def mlir_cmake_dir() -> Path:
    """``MLIR_DIR`` for ``find_package(MLIR CONFIG)`` in out-of-tree package builds."""
    return mlir_install() / "lib" / "cmake" / "mlir"


def available() -> bool:
    """True iff the install exposes the tools + cmake export an OOT C++ package needs."""
    return (
        mlir_bin("mlir-opt").is_file()
        and mlir_bin("mlir-translate").is_file()
        and mlir_bin("mlir-tblgen").is_file()
        and (mlir_cmake_dir() / "MLIRConfig.cmake").is_file()
    )


def require() -> Path:
    """Return the install prefix or raise a clear error naming the override env var."""
    if not available():
        raise RuntimeError(
            f"MLIR install not found/usable at {mlir_install()} "
            f"(expected bin/mlir-opt|mlir-translate|mlir-tblgen + lib/cmake/mlir/MLIRConfig.cmake). "
            f"Build it (Phase 0) or set MERLIN_MLIR_INSTALL."
        )
    return mlir_install()
