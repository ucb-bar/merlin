"""Explicit file compilation through a selected provider's public native APIs.

This closed workflow composes preparation, provider rewriting, native lowering
and ordinary object codegen. It does not reinterpret optional runner features,
infer a board, or select a profitable transformation automatically.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be an explicit boolean")
    return value


@dataclass(frozen=True)
class ProviderConfig:
    """No option is selected by default; geometry is supplied by the caller."""

    target: str
    vlen_bits: int
    elen_bits: int
    operand_bits: int
    accumulator_bits: int
    operand_lmul: int
    workspace_bytes: int
    stream_im2col: bool = False
    normalize_windows: bool = False
    canonicalize_contractions: bool = False
    exact_register_blocks: bool = False
    copy_loops: bool = False
    copy_max_rank: int = 4
    register_budget: int = 32
    reserved_registers: int = 4
    max_mr: int = 4
    cache_budget_bytes: int | None = None
    timeout_seconds: int = 120
    matrix: dict | None = None
    _input_binding: tuple[str, str] | None = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def load(cls, path: str | Path):
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict) or data.pop("schema", None) != "merlin.provider_workflow.v1":
            raise ValueError("provider config requires merlin.provider_workflow.v1")
        result = cls(**data)
        selected_path = Path(path).absolute()
        object.__setattr__(result, "_input_binding", (str(selected_path), sha(selected_path)))
        return result

    def __post_init__(self):
        if not isinstance(self.target, str) or not self.target.strip():
            raise ValueError("explicit support target required")
        for name in (
            "stream_im2col",
            "normalize_windows",
            "canonicalize_contractions",
            "exact_register_blocks",
            "copy_loops",
        ):
            _boolean(getattr(self, name), name)
        if self.stream_im2col and self.exact_register_blocks:
            raise ValueError("select streaming or exact-divisor register scheduling separately")
        for name in (
            "vlen_bits",
            "elen_bits",
            "operand_bits",
            "accumulator_bits",
            "operand_lmul",
            "workspace_bytes",
            "copy_max_rank",
            "register_budget",
            "reserved_registers",
            "max_mr",
            "timeout_seconds",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 1 <= self.copy_max_rank <= 8 or not 0 < self.timeout_seconds <= 86400:
            raise ValueError("invalid copy rank or compiler timeout")
        if self.cache_budget_bytes is not None and type(self.cache_budget_bytes) is not int:
            raise ValueError("cache budget must be an explicit integer")
        if self.matrix is not None:
            allowed = {"support_target", "unit", "config", "kernel", "streaming_options"}
            if not isinstance(self.matrix, dict) or set(self.matrix) - allowed:
                raise ValueError("unknown matrix selection fields")
            if any(
                not isinstance(self.matrix.get(k), str) or not self.matrix[k]
                for k in ("support_target", "unit", "config")
            ):
                raise ValueError("matrix selection requires support_target, unit and config")
            if self.matrix["support_target"] != self.target:
                raise ValueError("matrix and vector provider must share the selected support chain")


def _public_api(backend, name):
    factory = getattr(backend, name, None)
    if not callable(factory):
        raise ValueError(f"selected provider lacks public {name}; no fallback backend is permitted")
    return factory()


def _provider_sources(roots):
    """Observe loaded executable provider sources; this is not hermetic closure."""
    result = {}
    for module in tuple(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if file and any(Path(file).resolve().is_relative_to(root) for root in roots):
            path = Path(file)
            if path.is_file() and path.suffix == ".py":
                result[str(path.resolve())] = sha(path)
    return result


def _core_sources(provider_roots):
    """Refuse accidental namespace fallback to a different Merlin checkout."""
    root = Path(__file__).resolve().parents[1]
    result = {}
    for name, module in tuple(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if not file or not (name == "merlin" or name.startswith("merlin.")):
            continue
        path = Path(file).resolve()
        if any(path.is_relative_to(selected) for selected in provider_roots):
            continue
        if not path.is_relative_to(root):
            raise ValueError(f"mixed Merlin source roots are not a reproducible compiler invocation: {path}")
        if path.is_file():
            result[str(path)] = sha(path)
    return result


def _integer_source(module):
    """Keep floating-effect/runner policies on the ordinary lowering path."""
    from xdsl.dialects.builtin import AnyFloat

    for operation in module.walk():
        values = [*operation.operands, *operation.results]
        for region in operation.regions:
            values.extend(argument for block in region.blocks for argument in block.args)
        for value in values:
            element = getattr(value.type, "get_element_type", lambda: value.type)()
            if isinstance(element, AnyFloat):
                raise ValueError(
                    "provider-native workflow accepts typed integer IR; use ordinary lowering for floating policies"
                )


def compile_file(source, work, config: ProviderConfig, *, targets=()):
    """Compile supplied typed IR with explicit provider choices, retaining refusals.

    The ordinary lower command remains unchanged when this API is not selected.
    Runner-gated features and parallel dispatch are deliberately a separate API;
    they cannot be silently discarded by this closed provider-native workflow.
    """
    from . import toolchain
    from .backend_spi import load_vector_codegen
    from .codegen import build_host_shared, compile_ll
    from .lower import LowerResult
    from .passes_xdsl import preprocess_text

    source, work = Path(source).resolve(strict=True), Path(work)
    if not isinstance(config, ProviderConfig):
        raise TypeError("typed ProviderConfig required")
    if set(targets) - {"host", "riscv"}:
        raise ValueError("unknown object target")
    work.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "merlin.provider_workflow_receipt.v1",
        "status": "failed",
        "config": {k: v for k, v in asdict(config).items() if not k.startswith("_")},
        "source": str(source),
        "source_sha256": sha(source),
        "timing_certified": False,
        "physical_utilization_measured": False,
        "hermetic_dependency_closure": False,
        "commands": [],
    }
    inputs = {str(source): sha(source), str(Path(__file__).resolve()): sha(__file__)}
    objects = []
    try:
        if config._input_binding is not None:
            name, digest = config._input_binding
            if sha(name) != digest or ProviderConfig.load(name) != config:
                raise ValueError("provider config changed after selection")
            inputs[name] = digest
        backend = load_vector_codegen(config.target)
        stream_api = _public_api(backend, "im2col_stream_api")
        copy_api = _public_api(backend, "copy_loops_api")
        from merlin.targetgen.plugins import load_declared_chain

        declarations = load_declared_chain(config.target, "rvv_codegen")
        if config.matrix is not None:
            declarations += load_declared_chain(config.target, "matrix_lowering")
        roots = tuple(Path(inspect.getfile(module)).resolve().parent for _, module in declarations)
        inputs.update(_provider_sources(roots))
        inputs.update(_core_sources(roots))
        target = stream_api.PanelTarget(
            **{
                k: getattr(config, k)
                for k in (
                    "vlen_bits",
                    "elen_bits",
                    "operand_bits",
                    "accumulator_bits",
                    "operand_lmul",
                    "workspace_bytes",
                    "register_budget",
                    "reserved_registers",
                    "max_mr",
                    "cache_budget_bytes",
                )
            }
        )
        upstream, stats = preprocess_text(source.read_text())
        from merlin.common import mlir_query as mq

        _integer_source(mq.parse(upstream))
        observed_upstream = work / "model.upstream.mlir"
        observed_upstream.write_text(upstream)
        receipt["preprocessing"] = {"path": str(observed_upstream), "sha256": sha(observed_upstream)}
        prepared = work / "model.prepared.mlir"
        prepared.write_text(upstream)
        matrix = config.matrix
        integer_api = None
        if config.normalize_windows or config.canonicalize_contractions or config.exact_register_blocks:
            if matrix is not None:
                raise ValueError("matrix provider owns its source normalization; do not select normalize_windows")
            integer_api = _public_api(backend, "integer_file_api")
            prepared, integer_report = integer_api.prepare_file(
                prepared,
                work,
                target,
                windows=config.normalize_windows,
                contractions=config.canonicalize_contractions,
                blocks=config.exact_register_blocks,
            )
            receipt["integer_preparation"] = integer_report
            receipt["source_normalization"] = integer_report["normalization"]
        if matrix is not None:
            from merlin.runtime.backends.zephyr_model import MatrixRouting

            routing = MatrixRouting(
                matrix["support_target"], matrix["unit"], matrix["config"], kernel=matrix.get("kernel")
            )
            provider = routing.provider()
            edge, _ = provider.geometry(unit=routing.unit, config=routing.config)
            if config.stream_im2col:
                factory = getattr(provider, "streaming_selector", None)
                if not callable(factory):
                    raise ValueError("selected matrix provider has no streaming_selector")
                options = dict(matrix.get("streaming_options") or {})
                if set(options) - {"panel_rows", "prepacked_a", "admission"}:
                    raise ValueError("unknown matrix streaming option")
                selection = factory(
                    unit=routing.unit,
                    config=routing.config,
                    vlen_bits=config.vlen_bits,
                    workspace_bytes=config.workspace_bytes,
                    **options,
                )
            else:
                if matrix.get("streaming_options"):
                    raise ValueError("matrix streaming options require stream_im2col")
                selection = provider.selector(edge)
            rewrite = provider.rewrite_prepared_file(prepared, work, select=selection, tile_edge=edge)
            receipt["matrix_rewrite"] = rewrite.to_dict()
            schedule_text = backend.baseline_schedule_text()
        else:
            selected, report = stream_api.rewrite_prepared_file(prepared, target, work, enabled=config.stream_im2col)
            prepared = Path(selected)
            receipt["stream_rewrite"] = asdict(report)
            schedule_text = (
                stream_api.lowering_schedule(report) if report.streamed else backend.baseline_schedule_text()
            )
            if config.exact_register_blocks:
                schedule_text = integer_api.lowering_schedule(integer_report["register_blocks"]["table"])
        schedule = work / "provider-schedule.mlir"
        schedule.write_text(schedule_text)
        pipeline = (
            stream_api.lowering_pipeline(schedule)
            if matrix is None and (report.streamed or config.exact_register_blocks)
            else backend.build_pipeline(schedule, features=frozenset())
        )
        # These APIs have no callback channel for shared runner pseudo-passes.
        # A provider must give a complete native pipeline, never a silently lost gate.
        from merlin.compiler.lower.llvm.passes import finalize_pipeline

        pipeline = finalize_pipeline(pipeline)
        mlir_opt = Path(toolchain.llvm_install()) / "bin" / "mlir-opt"
        translated = Path(toolchain.mlir_translate())
        lowered = copy_api.lower_file(
            prepared,
            pipeline=pipeline,
            mlir_opt=mlir_opt,
            compiler_python=toolchain.compiler_python(),
            work=work / "provider-lowering",
            enabled=config.copy_loops,
            max_rank=config.copy_max_rank,
            timeout=config.timeout_seconds,
        )
        receipt["native_lowering"] = lowered
        ll = work / "model.ll"
        command = [str(translated), "--mlir-to-llvmir", lowered["llvm_dialect"], "-o", str(ll)]
        receipt["commands"].append(command)
        proc = subprocess.run(command, capture_output=True, timeout=config.timeout_seconds)
        (work / "translate.stdout").write_bytes(proc.stdout)
        (work / "translate.stderr").write_bytes(proc.stderr)
        if proc.returncode or not ll.is_file():
            raise RuntimeError(f"selected mlir-translate failed: {proc.returncode}")
        flags = backend.vector_cflags(["-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin"])
        from merlin.runtime.backends.zephyr_model import march_with_vlen

        flags = march_with_vlen(list(flags), config.vlen_bits)
        if matrix is not None:
            signatures = provider.load_signatures(work)
            receipt["matrix_signatures"] = signatures
            if signatures:
                from merlin.kernels.matrix_kernels import build_kwargs

                kernel_selection = build_kwargs(routing, signatures)
                built = provider.build_object(
                    signatures,
                    work / "matrix",
                    unit=routing.unit,
                    config=routing.config,
                    cc=toolchain.clang(),
                    cflags=["--target=riscv64-unknown-elf", *flags],
                    scalar_tile=False,
                    parallel_tiles=False,
                    **kernel_selection,
                )
                receipt["matrix_object"] = built.to_dict()
                objects.append(str(Path(built.object_path).resolve()))
        stats["provider_selection"] = {
            "normalized_contractions": receipt.get("source_normalization", {}).get("normalized", 0),
            "streamed": receipt.get("stream_rewrite", {}).get("streamed")
            if matrix is not None
            else receipt["stream_rewrite"]["streamed"],
            "copy_rewrites": lowered["copy_report"]["rewritten"],
            "matrix_routed": receipt.get("matrix_rewrite", {}).get("count", 0),
        }
        result = LowerResult(work, ll, stats={**stats, "provider_workflow": str(work / "provider-workflow.json")})
        if "host" in targets:
            if objects:
                raise ValueError(
                    "matrix-linked objects are target-specific; host execution is a separate numerical gate"
                )
            result.host_so = build_host_shared(ll, work / "model_host.so")
        if "riscv" in targets:
            result.riscv_obj = compile_ll(
                ll, work / "model_riscv.o", "riscv", extra_flags=tuple(flags), timeout_s=config.timeout_seconds
            )
        for api in (stream_api, copy_api):
            path = Path(inspect.getfile(api)).resolve()
            inputs[str(path)] = sha(path)
        for path, digest in _provider_sources(roots).items():
            inputs.setdefault(path, digest)
        for path, digest in _core_sources(roots).items():
            inputs.setdefault(path, digest)
        for path, digest in inputs.items():
            if sha(path) != digest:
                raise RuntimeError(f"workflow input changed: {path}")
        receipt.update(
            status="compiled",
            input_sha256=inputs,
            additional_objects=objects,
            additional_object_sha256={p: sha(p) for p in objects},
            cflags=flags,
            llvm_sha256=sha(ll),
            object_sha256=sha(result.riscv_obj) if result.riscv_obj else None,
            prepared_sha256=sha(prepared),
            selected_prepared=str(prepared),
        )
        return result
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (work / "provider-workflow.json").write_text(json.dumps(receipt, indent=2, default=str) + "\n")
