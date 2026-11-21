"""Program-local, immutable-input preparation for ordinary host admission.

Only declaration validation and selector lookup are retained. Every supplied
source body and every admission decision is still checked by the original
consumer; this object is neither a source proof nor a persistent verdict cache.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import functools
import hashlib
import inspect
import json
import os
import pickle
import subprocess
import sys
import uuid
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from merlin.common.digest import is_sha256
from merlin.common.process import invocation_records as invocation_record
from merlin.compiler.lower.host.host_optimization_request import (
    POLICY_SCHEMA,
    HostOptimizationSelection,
    context_from_software,
    request_from_record,
)
from merlin.compiler.lower.model.link_supplier_trace import trace_symbol_flags
from merlin.target.contract.interfaces.linalg import bind_logical_io_metadata, is_linalg_on_tensors

# ---- host_admission_preparation ----------------------------------------------------------------


def _bytes(selected):
    return json.dumps(selected, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


_ISSUED = {}


class PreparedHostCapabilities(Mapping):
    """Own a private copy and an order-preserving exact-selector index."""

    def __init__(self, selected):
        from merlin.development.targetgen.policies.host_capabilities import validate_host_capabilities

        if not isinstance(selected, dict):
            raise ValueError("selected host capabilities must be a profile mapping")
        input_bytes = _bytes(selected)
        selected_copy = copy.deepcopy(selected)
        indices = {}
        for name, selection in sorted(selected_copy.items()):
            if not isinstance(selection, dict):
                raise ValueError("selected host capability profile must be a mapping")
            document = selection.get("capability_spec")
            if document is None:
                continue
            validate_host_capabilities(
                document, package_sha256=selection.get("package_sha256"), dtype_strategy=selection.get("dtype_strategy")
            )
            exact, fallback = {}, []
            for position, declaration in enumerate(document["operations"]):
                selectors = declaration.get("ops") or []
                if not selectors:
                    fallback.append(position)
                for operation in selectors:
                    exact.setdefault(operation, set()).add(position)
            indices[name] = exact, tuple(fallback)
        identity = id(self)
        _ISSUED[identity] = (
            weakref.ref(self, lambda _: _ISSUED.pop(identity, None)),
            input_bytes,
            selected_copy,
            indices,
        )
        self.require_current(selected)

    def _binding(self):
        original = _ISSUED.get(id(self))
        if original is None or original[0]() is not self:
            raise ValueError("host capability profile has no original issued preparation")
        return original

    def __len__(self):
        return len(self._binding()[2])

    def __iter__(self):
        return iter(self._binding()[2])

    def __getitem__(self, name):
        # Other consumers cannot mutate the prepared document through Mapping.
        return copy.deepcopy(self._binding()[2][name])

    def require_current(self, selected):
        """Reopen the caller's selected bytes before accepting the program."""
        _, input_bytes, selected_copy, _ = self._binding()
        if _bytes(selected) != input_bytes or selected != selected_copy:
            raise ValueError("selected host capability profile changed during source admission")

    def _items(self):
        return sorted(self._binding()[2].items())

    def _declarations(self, name, row):
        _, _, selected, indices = self._binding()
        exact, fallback = indices[name]
        positions = set(fallback)
        for identity in (row.get("frontend_op"), row["mlir_operation"]):
            positions.update(exact.get(identity, ()))
        document = selected[name]["capability_spec"]
        return tuple(document["operations"][position] for position in sorted(positions))


# ---- host_linkage_contract ---------------------------------------------------------------------
# Closed, opt-in host library-supplier requirement; never a numerical grant.

SCHEMA = "merlin.host_linkage_contract.v1"
SUPPLIER = "selected_math_archive"
RUNTIME_SCHEMA = "merlin.host_runtime_linkage_contract.v1"
RUNTIME_SUPPLIER = "selected_host_math_runtime_object"


def runtime_recipe(contract: Mapping[str, Any], policy: str) -> dict[str, Any]:
    """Match an explicit runtime selection to reviewed source and strict flags."""
    from hashlib import sha256

    from merlin.runtime.host_math import host_math_recipe

    selected = [row for row in contract["recipes"] if row["policy"] == policy]
    recipe = host_math_recipe(policy)
    actual = {
        "policy": policy,
        "source_sha256": sha256(recipe.source.encode()).hexdigest() if recipe.source is not None else None,
        "compile_flags": list(recipe.compile_flags),
        "link_flags": list(recipe.link_flags),
    }
    if len(selected) != 1 or selected[0] != actual:
        raise ValueError("reviewed runtime supplier differs from selected host math recipe/source/flags")
    wrappers = {flag.removeprefix("-Wl,--wrap=") for flag in recipe.link_flags if flag.startswith("-Wl,--wrap=")}
    if not set(contract["symbols"]) <= wrappers:
        raise ValueError("reviewed runtime supplier does not wrap every selected source symbol")
    return actual


def _runtime_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    if (
        set(value) != {"schema", "supplier", "symbols", "recipes"}
        or value.get("schema") != RUNTIME_SCHEMA
        or value.get("supplier") != RUNTIME_SUPPLIER
        or not isinstance(value.get("symbols"), list)
        or not isinstance(value.get("recipes"), list)
        or not 1 <= len(value["recipes"]) <= 16
    ):
        raise ValueError("host linkage contract is not a closed selected runtime roster")
    symbols = list(value["symbols"])
    trace_symbol_flags(symbols)
    if symbols != sorted(symbols):
        raise ValueError("host linkage symbols must be in canonical order")
    recipes = []
    for row in value["recipes"]:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"policy", "source_sha256", "compile_flags", "link_flags"}
            or not isinstance(row.get("policy"), str)
            or not row["policy"]
            or not is_sha256(row.get("source_sha256"))
            or any(
                not isinstance(row.get(key), list)
                or not row[key]
                or any(not isinstance(flag, str) or not flag for flag in row[key])
                for key in ("compile_flags", "link_flags")
            )
        ):
            raise ValueError("selected runtime recipe identity is incomplete")
        recipes.append({key: list(item) if isinstance(item, list) else item for key, item in row.items()})
    policies = [row["policy"] for row in recipes]
    if policies != sorted(set(policies)):
        raise ValueError("selected runtime policies must be unique and canonically ordered")
    return {"schema": RUNTIME_SCHEMA, "supplier": RUNTIME_SUPPLIER, "symbols": symbols, "recipes": recipes}


def validate_linkage_contract(value: object) -> dict[str, Any]:
    """Require an exact caller-selected archive digest and bounded symbol roster."""
    if isinstance(value, Mapping) and value.get("supplier") == RUNTIME_SUPPLIER:
        return _runtime_contract(value)
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema", "supplier", "archive_sha256", "symbols"}
        or value.get("schema") != SCHEMA
        or value.get("supplier") != SUPPLIER
        or not is_sha256(value.get("archive_sha256"))
        or not isinstance(value.get("symbols"), list)
    ):
        raise ValueError("host linkage contract is not a closed selected archive roster")
    symbols = list(value["symbols"])
    trace_symbol_flags(symbols)
    if symbols != sorted(symbols):
        raise ValueError("host linkage symbols must be in canonical order")
    return {"schema": SCHEMA, "supplier": SUPPLIER, "archive_sha256": value["archive_sha256"], "symbols": symbols}


def required_composite_math_symbol(schema: object, operation: object) -> str | None:
    """Name an unresolved selected-library obligation, not an observed call."""
    from merlin.compiler.frontend.patterns.linalg_composite_math import (
        STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA,
    )
    from merlin.compiler.frontend.patterns.linalg_mixed_patterns import SCHEMA as MIXED_SCHEMA

    if schema == MIXED_SCHEMA and type(operation) is str:
        return {"f64_math_sin": "sin", "f64_math_cos": "cos", "f64_arith_constant_math_powf": "pow"}.get(operation)
    if schema != STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA or type(operation) is not str:
        return None
    return {"literal_base_pow_f32": "powf", "tanh_gelu_f32": "tanhf"}.get(operation)


def validate_source_linkage_contract(schema: object, operation: object, value: object) -> dict[str, Any]:
    """Restrict new composite forms to their exact selected symbol requirement."""
    from merlin.compiler.frontend.patterns.linalg_composite_math import (
        STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA,
    )
    from merlin.compiler.frontend.patterns.linalg_math_patterns import STATIC_F32_MATH_SOURCE_BODY_SCHEMA

    contract = validate_linkage_contract(value)
    if contract["supplier"] == RUNTIME_SUPPLIER:
        unary = {"math.sin": "sinf", "math.cos": "cosf", "math.exp": "expf"}
        symbol = (
            unary.get(operation)
            if schema == STATIC_F32_MATH_SOURCE_BODY_SCHEMA and type(operation) is str
            else required_composite_math_symbol(schema, operation)
        )
        if symbol not in {"sinf", "cosf", "expf", "powf", "tanhf"} or contract["symbols"] != [symbol]:
            raise ValueError("runtime linkage contract does not match the exact f32 math source form")
        return contract
    if (
        schema == STATIC_F32_MATH_SOURCE_BODY_SCHEMA
        and type(operation) is str
        and operation in {"math.sin", "math.cos"}
    ):
        return contract  # Preserve the previously reviewed unary-math contract.
    symbol = required_composite_math_symbol(schema, operation)
    from merlin.compiler.frontend.patterns.linalg_mixed_patterns import SCHEMA as MIXED_SCHEMA

    if (
        schema not in {STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA, MIXED_SCHEMA}
        or symbol is None
        or contract["symbols"] != [symbol]
    ):
        raise ValueError("linkage contract does not match the exact composite math source form")
    return contract


# ---- host_optimization_admission ---------------------------------------------------------------
# Original reviewed host permissions and source-bound command-buffer choices.
#
# The request is candidate data. Permissions come only from the original software
# file already sealed in the common profile. This is not a numerical, physical,
# Lean or deployment qualification. Unselected builds retain their existing path.

_ACTIVE = contextvars.ContextVar("original_host_optimization_selection", default=None)


def active_selection():
    return _ACTIVE.get()


def _original_entry(admission, target, build_service):
    if build_service is not None:
        from merlin.target.contract.build.services import BuildOnlyService

        if type(build_service) is not BuildOnlyService:
            raise ValueError("host choices require the original typed build service")
        build_service.verify(target)
        return build_service.recipe.require_kernel_stack_frame().entry_symbol, None
    from merlin.common.storage.bundle_snapshots import snapshot_input_paths
    from merlin.target.contract.interfaces.abi import from_contract
    from merlin.target.registry.identity import resolve

    # Ordinary operator oracles use the selected capability contract rather
    # than a BuildOnlyService. Require that exact original input, with no new
    # grant, backend callback or mutable provider fallback.
    selected = resolve(target)
    path = selected.capability_contract_path.absolute()
    [frozen] = snapshot_input_paths(admission.workspace, admission.bundle, [path], repo=admission.repo)
    payload = frozen.read_bytes()
    contract = yaml.safe_load(payload)
    if type(contract) is not dict or contract.get("name") != target or selected.load_contract() != contract:
        raise ValueError("host choices changed the original selected capability contract")
    return from_contract(contract, target=target).entry_symbol, (str(path), hashlib.sha256(payload).hexdigest())


def _original_context(admission, entry):
    from merlin.common.storage import bundle_snapshots as bwrap
    from merlin.development.targetgen.authoring.tool_profile import require_current_substrate

    admission.verify()
    require_current_substrate(admission.profile["common_substrate"])
    row = admission.profile["semantic_inputs"]["software_spec"]
    if row is None:
        raise ValueError("host choices have no original reviewed software permissions")
    [source] = bwrap.snapshot_input_paths(
        admission.workspace, admission.bundle, [Path(row["path"])], repo=admission.repo
    )
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != row["sha256"]:
        raise ValueError("host permissions differ from the original software bytes")
    software = yaml.safe_load(payload)
    return context_from_software(
        software,
        entry=entry,
        semantic_sha256=row["sha256"],
        target=admission.profile["semantic_inputs"]["target"],
        index_bits=admission.profile["host_index_bits"],
    )


@contextlib.contextmanager
def selected_buffer_choices(buffer, source, *, target, build_service=None):
    """Bind a request to one actual artifact under a live original admission."""
    record = buffer.get("host_optimization_request")
    if "host_optimization_request" not in buffer:
        yield None
        return
    from merlin.development.targetgen.authoring.profile_admission import active_compiler_imports

    if _ACTIVE.get() is not None:
        raise ValueError("host choices cannot replace an active original selection")
    request_bytes = json.dumps(record, sort_keys=True, allow_nan=False).encode()
    if len(request_bytes) > 256 * 1024:
        raise ValueError("host choice request exceeds the original transport budget")
    request = request_from_record(record)
    if hashlib.sha256(source.encode()).hexdigest() != request.source_sha256:
        raise ValueError("host choice request does not bind the actual terminal artifact")
    admission = active_compiler_imports()
    if admission is None or admission.profile["semantic_inputs"]["target"] != target:
        raise ValueError("host choices require the matching original compiler admission")
    entry, original_entry = _original_entry(admission, target, build_service)
    selection = HostOptimizationSelection(_original_context(admission, entry), request)
    # Recheck compatibility before any legacy build cache could skip the producer.
    from merlin.compiler.lower.host.host_optimization_request import prepare_host_optimization

    prepare_host_optimization(source, selection, finished_llvm_only=build_service is not None)
    token = _ACTIVE.set(selection)
    try:
        yield selection
        admission.verify()
        current = json.dumps(buffer.get("host_optimization_request"), sort_keys=True, allow_nan=False).encode()
        if current != request_bytes:
            raise ValueError("host choice request changed during its actual build")
        if _original_context(admission, entry) != selection.context:
            raise ValueError("original host permissions changed during compilation")
        if _original_entry(admission, target, build_service) != (entry, original_entry):
            raise ValueError("host choices changed their original selected entry during compilation")
    finally:
        _ACTIVE.reset(token)


def host_choices_build(function):
    """Keep normal numerical/link builds on the same original scoped selection."""

    @functools.wraps(function)
    def build(buffer, source, *args, **kwargs):
        with selected_buffer_choices(
            buffer, source, target=kwargs["target"], build_service=kwargs.get("_build_service")
        ):
            return function(buffer, source, *args, **kwargs)

    return build


# ---- input_preparation -------------------------------------------------------------------------
# Bind optional input-copy auditing to a capsule's original logical convolution ABI.
#
# This supplies no kernel or numerical reference. The independently selected harness determines the
# pointer order; the original program determines types and extents. The emitted code must still
# prove its actual copies, bounds and accelerator consumption.


def source_convolution(capsule, *, target):
    from merlin.compiler.lower.device.shim import kernel_abi_for
    from merlin.runtime.backends.registry import harness_build_recipe
    from merlin.target.contract.harness.harness_render import logical_abi, logical_interface
    from merlin.target.contract.interfaces.grammar import parse_interface_mlir

    if (capsule.get("operation") or {}).get("op") != "conv2d":
        return None
    kernel_abi = kernel_abi_for(target)
    if kernel_abi is None or kernel_abi.version != 2:
        return None
    name = capsule.get("interface_mlir")
    if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("input preparation requires a contained original interface")
    source = Path(capsule["__dir__"]) / name
    if any(path.is_symlink() for path in (source, *source.parents)):
        raise ValueError("input preparation original source is indirect")
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    screen = capsule.get("software_screen") or {}
    if screen.get("status") != "admitted" or screen.get("program_sha256") != digest:
        raise ValueError("input preparation lacks original source admission")
    original = parse_interface_mlir(raw.decode("utf-8"))
    commands = original["commands"]
    if original["target"] != target or [row["opcode"] for row in commands] != ["RES_PACK", "CONV2D", "EVICT"]:
        return None
    pack, convolution, evict = commands
    attrs = convolution.get("attributes") or {}
    if attrs.get("epilogue") or attrs.get("output_dtype") != "i32":
        return None
    weights, handle = pack["operands"]["src"], pack["operands"]["dst"]
    if convolution["operands"]["weight"] != handle or evict["operands"]["handle"] != handle:
        raise ValueError("original convolution has an unbound resident weight")
    roles = {convolution["operands"]["ifm"]: "input_0", weights: "weight_0", convolution["operands"]["dst"]: "out_0"}
    abi = logical_abi()
    buffers = logical_interface(original, abi)
    if len(roles) != 3 or len(buffers) != 3 or {buffer.name for buffer in buffers} != set(roles):
        raise ValueError("original convolution lacks an exact three-buffer ABI")
    context = [{"role": roles[buffer.name], "shape": list(buffer.shape), "dtype": buffer.dtype} for buffer in buffers]
    if [buffer["role"] for buffer in context] != ["input_0", "weight_0", "out_0"]:
        return None
    # The logical ABI fixes buffer order, not the selected executable's entry.
    # Use the same independently selected recipe as the native compiler and
    # stack preflight. A candidate's own function spelling grants no selection.
    try:
        entry = harness_build_recipe(target).require_kernel_stack_frame().entry_symbol
    except (KeyError, NotImplementedError) as error:
        raise ValueError("input preparation has no selected harness entry") from error
    return {
        "source_sha256": digest,
        "entry_symbol": entry,
        "call_buffers": context,
        "argument_alignment": abi.alignment_bytes,
    }


# ---- public_source_abi -------------------------------------------------------------------------
# Observe neutral public ABI metadata enrichment before ordinary lowering.
#
# The selected capsule remains the source/numerical authority. A derived source
# adds only checked function attributes. Original and normalized source, complete
# public metadata and the actual helper call stay pinned through candidate use.
# The existing emission observer consumes the exact normalized bytes unchanged;
# no saved receipt can select source bytes or bypass its ordinary checks.
#
# :func:`compiler_source` carries the original bytes, their hash and the exact
# bytes the compiler receives together. A source-identity check (the mixed-program
# plan's ``source_sha256``) then compares the compiler's claim with the bytes it
# was given, while those bytes are re-derived here from the selected original and
# never read back from a generated copy.

_KEYS = ("label", "interface_mlir", "inputs", "component_program")


@dataclass(frozen=True)
class CompilerSource:
    """The selected original source and the exact bytes a compiler receives for it."""

    original_bytes: bytes
    compiler_bytes: bytes

    @property
    def original_sha256(self) -> str:
        return hashlib.sha256(self.original_bytes).hexdigest()

    @property
    def compiler_sha256(self) -> str:
        return hashlib.sha256(self.compiler_bytes).hexdigest()

    @property
    def enriched(self) -> bool:
        return self.compiler_bytes != self.original_bytes


@dataclass(frozen=True)
class _Selection:
    original: Path
    metadata: Path
    metadata_bytes: bytes
    declared: dict
    text: str
    inputs: list
    outputs: list


def _selection(capsule, source: Path, raw: bytes) -> _Selection | None:
    """The checked public selection, or None when the source passes through unchanged."""
    program = capsule.get("component_program")
    if (
        capsule.get("label") != "public"
        or not isinstance(program, dict)
        or program.get("schema") != "merlin.component_program.v2"
    ):
        return None
    text = raw.decode("utf-8")
    if not is_linalg_on_tensors(text):
        return None
    original = source.resolve(strict=True)
    if source.is_symlink():
        raise ValueError("public ABI source cannot be a symlink")
    root = Path(capsule["__dir__"]).resolve(strict=True)
    metadata = root / "capsule.yaml"
    if metadata.is_symlink() or not original.is_relative_to(root):
        raise ValueError("public ABI metadata/source escapes its selected capsule")
    if metadata.stat().st_size > 2 << 20 or original.stat().st_size > 16 << 20:
        raise ValueError("public ABI enrichment exceeds its bounded input size")
    metadata_bytes = metadata.read_bytes()
    declared = yaml.safe_load(metadata_bytes)
    if any(capsule.get(key) != declared.get(key) for key in _KEYS):
        raise ValueError("public ABI metadata differs from the original loaded capsule")
    if (root / declared.get("interface_mlir", "capsule.interface.mlir")).resolve(strict=True) != original:
        raise ValueError("public ABI interface differs from the selected source")
    inputs, outputs = declared.get("inputs"), declared["component_program"].get("outputs")
    if type(inputs) is not list or type(outputs) is not list:
        raise ValueError("public ABI has no complete ordered inputs/outputs")
    return _Selection(original, metadata, metadata_bytes, declared, text, inputs, outputs)


def _unchanged(selection: _Selection, raw: bytes) -> None:
    if selection.original.read_bytes() != raw or selection.metadata.read_bytes() != selection.metadata_bytes:
        raise ValueError("public ABI source/metadata changed before observed transformation")


def compiler_source(capsule, source) -> CompilerSource:
    """Re-derive the exact bytes :func:`public_abi_source` gives a compiler for ``source``.

    The same selection checks apply and the derivation is the same pure,
    body-preserving enrichment of the selected original's bytes, so a caller can
    bind a compiler's source claim to the bytes it actually received while the
    binding still proves derivation from the selected public source.
    """
    source = Path(source)
    raw = source.read_bytes()
    selection = _selection(capsule, source, raw)
    if selection is None:
        return CompilerSource(raw, raw)
    rendered, _ = bind_logical_io_metadata(selection.text, selection.inputs, selection.outputs)
    _unchanged(selection, raw)
    return CompilerSource(raw, rendered.encode("utf-8"))


@contextlib.contextmanager
def public_abi_source(capsule, source, generated):
    source = Path(source)
    raw = source.read_bytes()
    selection = _selection(capsule, source, raw)
    if selection is None:
        yield source
        return
    work = Path(generated).resolve() / "public-source-abi" / uuid.uuid4().hex
    work.mkdir(parents=True, mode=0o700)
    normalized = work / "interface.mlir"
    receipt = work / "source_binding.json"
    with invocation_record.observe_call(
        work,
        stage="public_logical_io_enrichment",
        function=bind_logical_io_metadata,
        arguments={"scope": "public ordered ABI names, roles and exact tensor types only"},
        inputs=(selection.original, selection.metadata),
        outputs=(normalized, receipt),
        dependencies=(Path(__file__), Path(inspect.getsourcefile(bind_logical_io_metadata))),
    ) as call:
        _unchanged(selection, raw)
        rendered, report = bind_logical_io_metadata(selection.text, selection.inputs, selection.outputs)
        report.update(
            original_source=str(selection.original),
            public_metadata=str(selection.metadata),
            public_metadata_sha256=hashlib.sha256(selection.metadata_bytes).hexdigest(),
        )
        normalized.write_text(rendered, encoding="utf-8")
        receipt.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        call.returned()
    invocation_record.verify(call.path)
    # The normal lower_interface observer continues to check source==staged
    # input and complete target/object custody. This outer owner additionally
    # keeps the source-to-metadata transition live across that invocation.
    try:
        yield normalized
    finally:
        invocation_record.verify(call.path)
        if any(capsule.get(key) != selection.declared.get(key) for key in _KEYS):
            raise ValueError("public ABI metadata changed during compiler execution")


# ---- host_compute_veto_process -----------------------------------------------------------------
# Evaluate the host-compute veto in a short-lived child process that imports the code on disk.
#
# A campaign's grader is one long-lived process: it imports the audit once at launch, so a corrected
# check never reached a background or final grade of a run already going, only fresh self-check
# processes. Each grade therefore runs the veto in a new interpreter (``python -m`` this module),
# resolving imports through the caller's ``sys.path`` and working directory, the same way grade
# offload workers start. The child applies exactly the in-process veto, writes the same result files,
# and hands back the updated rows, which the caller copies into its own row objects.
#
# The child cannot change what the veto may do: it can fail a pass, never manufacture one. If it
# cannot be started, dies, or its reply is unreadable, the veto runs in process as before.
# ``MERLIN_HOST_COMPUTE_VETO_PROCESS=0`` keeps it in process.

ENV = "MERLIN_HOST_COMPUTE_VETO_PROCESS"
#: Set in the child so a veto there never starts another child.
_CHILD_ENV = "MERLIN_HOST_COMPUTE_VETO_CHILD"


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip() != "0" and _CHILD_ENV not in os.environ


def run(results: list[dict[str, Any]], suite_runs: Path, *, capsules, target) -> list[dict[str, Any]] | None:
    """The vetoed rows, with ``results`` updated in place; ``None`` when the child gave no answer."""
    try:
        request = pickle.dumps((results, str(suite_runs), capsules, target), protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:  # noqa: BLE001 -- unpicklable rows stay in process
        return None
    env = dict(os.environ)
    env[_CHILD_ENV] = "1"
    # Resolve imports exactly as the caller does, including paths it added at run time.
    env["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
    try:
        stderr = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        stderr = None
    try:
        done = subprocess.run(
            [sys.executable, "-m", __name__],
            input=request,
            stdout=subprocess.PIPE,
            stderr=stderr,
            env=env,
            close_fds=True,
            check=False,
        )
    except (OSError, ValueError):
        return None
    if done.returncode != 0:
        return None
    try:
        updated, vetoed = pickle.loads(done.stdout)
    except Exception:  # noqa: BLE001 -- an unreadable reply: the in-process veto decides
        return None
    if not isinstance(updated, list) or len(updated) != len(results) or not isinstance(vetoed, list):
        return None
    for row, new in zip(results, updated, strict=True):
        if row != new:
            row.clear()
            row.update(new)
    return vetoed


def _serve() -> int:
    """Child: read one request on stdin, apply the veto, reply on the original stdout."""
    reply = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    # Anything the audit prints goes to stderr, so it cannot corrupt the reply.
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    results, suite_runs, capsules, target = pickle.loads(sys.stdin.buffer.read())
    from merlin.development.targetgen.conformance.offload_census import _apply_host_compute_veto_here

    vetoed = _apply_host_compute_veto_here(results, Path(suite_runs), capsules=capsules, target=target)
    reply.write(pickle.dumps((results, vetoed), protocol=pickle.HIGHEST_PROTOCOL))
    reply.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(_serve())
