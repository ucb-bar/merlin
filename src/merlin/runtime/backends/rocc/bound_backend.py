"""Contract-selected execution of prebuilt RoCC simulators (:class:`BoundRoCCBackend`)."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from merlin.common.process import invocation_records as I
from merlin.runtime.backends.rocc.tool_selection import (
    _HEADERS,
    _RUNTIME,
    _SCOPE,
    RoCCExecutionError,
    _bounded_process,
    _check_pin,
    _digest,
    _execution_binding,
    _execution_timeout_ceiling,
    _gsim_budget_max_cycles,
    _gsim_startup_transport,
    _model_spike_timeout_ceiling,
    _path,
    _require,
    _resource,
    _validate_engine,
)
from merlin.target.contract.build.services import HarnessBuildRecipe, KernelStackFramePolicy
from merlin.target.contract.harness.readback_policy import (
    COHERENT_DUMP_V1,
    COHERENT_PACKET_V1,
    COHERENT_PACKET_V2,
    FULL_VALUES_B64,
    ReadbackPolicy,
)

# ---- rocc_bound_backend ------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundRoCCBackend:
    target: str
    _selection_json: str
    rocc_semantics: object
    __name__ = __name__
    __file__ = __file__
    EXECUTION_CAPABILITIES = {
        "whole_program_kernel_abi": "core logical pointer harness with original full input/output membership"
    }
    REQUIRES_EXPLICIT_LOGICAL_INPUTS = True

    def _data(self):
        return json.loads(self._selection_json)

    def _block(self):
        return self._data()["contract"]["runner"]["chipyard_rocc"]

    def operand_source_relocations(self, original_facts_path):
        """Reopen the original selected observer's byte-identical source moves."""
        from merlin.target.isa.rocc.operand_protocol import selected_source_relocations
        from merlin.target.isa.rocc.semantics import BoundSemantics

        data = self._data()
        selection = data["contract"].get("rocc_operand_selection")
        if selection is None:
            return None
        path = Path(original_facts_path)
        _require(path.is_file() and not path.is_symlink(), "selected original facts are missing or indirect")
        _require(_digest(path) == selection["original_facts"]["sha256"], "execution uses different original facts")
        original = BoundSemantics(target=self.target, contract=data["contract"], facts=data["facts"])
        return selected_source_relocations(original, selection)

    def _execution_firrtl_sha256(self):
        """The FIRRTL an engine of the runner's configuration must be built from (see :func:`bind`)."""
        return _execution_binding(self._data()["facts"], self._block()["config"])["firrtl"]["sha256"]

    @property
    def ORACLE(self):
        return {
            engine: {
                "engine": engine,
                "kind": "functional" if engine == "spike" else "rtl",
                "fidelity": "functional_model" if engine == "spike" else "elaborated_rtl",
                "derived_from_rtl": engine in {"gsim", "verilator"},
                "scope": _SCOPE,
            }
            for engine in self._block()["engines"]
        }

    def _readout(self, key):
        return self._block().get("readout", {}).get(key)

    def readback_policy(self):
        from merlin.runtime.harness.pointer_storage import validate_contract

        return validate_contract(self._data()["contract"]).readback_policy

    def engine_readback_policy(self, simulator, *, cb=None):
        """Only an explicit sealed engine declaration overrides the default."""
        block = self._engine(simulator)
        transport = block.get("readback_transport", self.readback_policy().transport)
        if transport in {COHERENT_DUMP_V1, COHERENT_PACKET_V1, COHERENT_PACKET_V2} and cb is not None:
            from merlin.runtime.harness.pointer_storage import _roster, _tensor

            normalized, _ = _roster(cb)
            if any(
                not _tensor(normalized["tensors"][name])[0]
                or normalized["tensors"][name]["dtype"] not in {"i8", "i16", "i32", "i64", "f32"}
                for name in normalized["kernel_abi"]["outputs"]
            ):
                return self.readback_policy()
        if transport in (COHERENT_PACKET_V1, COHERENT_PACKET_V2):
            required = {"htif", "out_b64", "out_bin", "out_bin_memory"}
            if transport == COHERENT_PACKET_V2:
                required.add("out_packet_v2")
            _require(
                required <= {row["id"] for row in self._block()["toolchain"]["headers"]},
                "selected packet transport omits its original complete codec header roster",
            )
        return ReadbackPolicy(transport)

    def memory_readback_transport(self, simulator, *, policy_transport=None):
        transport = self.engine_readback_policy(simulator).transport
        if policy_transport not in (None, transport) or transport not in {
            COHERENT_DUMP_V1,
            COHERENT_PACKET_V1,
            COHERENT_PACKET_V2,
        }:
            return None
        return "gsim_" + transport if simulator == "gsim" else None

    def readout_scalar_abi(self):
        return self._readout("scalar_abi")

    def readout_epilogue_capability(self):
        return self._readout("epilogue_capability")

    def readout_operand_sum(self):
        return self._readout("operand_sum")

    def epilogue_stage_routes(self):
        return self._readout("stage_routes") or []

    def harness_build_recipe(self):
        block = self._block()["toolchain"]
        compiler = _check_pin(block["compiler"], executable=True)
        script = _check_pin(block["link_script"])
        _require(script.stat().st_size <= 1024 * 1024, "selected linker layout exceeds the bounded input limit")
        with script.open("rb") as stream:
            text = stream.read(1024 * 1024 + 1)
        _require(len(text) <= 1024 * 1024, "selected linker layout exceeds the bounded input limit")
        words = "".join(c if c.isalnum() or c == "_" else " " for c in text.decode("utf-8")).split()
        _require(
            not {"INPUT", "GROUP", "SEARCH_DIR", "INCLUDE", "STARTUP"}.intersection(words),
            "selected linker layout includes external compilation inputs",
        )
        units, headers = [], []
        for rows, roster, dest in ((block["runtime_units"], _RUNTIME, units), (block["headers"], _HEADERS, headers)):
            for row in rows:
                path = _resource(row["id"], roster)
                _check_pin({"path": str(path), "sha256": row["sha256"]})
                dest.append(path)
        return HarnessBuildRecipe(
            compiler=compiler,
            include_roots=tuple(dict.fromkeys(p.parent for p in headers)),
            support_sources=tuple(units),
            header_dependencies=tuple(headers),
            link_script=script,
            load_address=block["load_address"],
            cflags=tuple(block["cflags"]),
            ldflags=tuple(block["ldflags"]),
            error_cls=RoCCExecutionError,
            kernel_stack_frame=KernelStackFramePolicy(**block["kernel_stack_frame"]),
        )

    def gcc_path(self):
        return _check_pin(self._block()["toolchain"]["compiler"], executable=True)

    def verify_execution_inputs(self):
        """Check selected build and engine identities without executing tools."""
        self.harness_build_recipe()
        return {engine: self._engine_identity(engine) for engine in self._block()["engines"]}

    def logical_storage_budget_bytes(self):
        from merlin.runtime.harness.pointer_storage import selected_storage_budget

        return selected_storage_budget(self._data()["contract"])

    def render_harness(
        self,
        cb,
        *,
        target,
        inputs=None,
        readback_policy=None,
        warm_profile=None,
        source_owned_mutables=(),
        blobs=None,
        host_storage_limit_bytes=None,
        prepack_authorizations=None,
    ):
        _require(target == self.target, "logical harness target differs from the selected backend")
        if warm_profile is not None:
            raise NotImplementedError("selected logical harness has no qualified warm profile transport")
        from merlin.runtime.harness.pointer_storage import render_harness

        contract = self._data()["contract"]
        # Preserve the existing explicit source-owned host limit when no
        # contract budget was selected. An ordinary caller still uses the
        # neutral renderer's unchanged default.
        budget = (contract.get("logical_harness") or {}).get("storage_budget_bytes")
        if budget is not None:
            budget = self.logical_storage_budget_bytes()
        if host_storage_limit_bytes is not None:
            _require(
                type(host_storage_limit_bytes) is int and 0 < host_storage_limit_bytes <= 256 << 20,
                "prospective logical host storage exceeds its fixed software ceiling",
            )
            budget = min(budget, host_storage_limit_bytes) if budget is not None else host_storage_limit_bytes
        return render_harness(
            cb,
            target=target,
            inputs=inputs,
            contract=contract,
            readback_policy=readback_policy,
            source_owned_mutables=source_owned_mutables,
            blobs=blobs,
            prepack_authorizations=prepack_authorizations,
            **({"max_storage_bytes": budget} if budget is not None else {}),
        )

    def complete_call_timing(self):
        from merlin.runtime.harness.call_timing import complete_call_plan as from_contract

        return from_contract(self._data()["contract"])

    def build_source_paths(self):
        from merlin.analysis.programs import storage_encoding
        from merlin.analysis.programs import storage_encoding as tensor_storage
        from merlin.runtime.engine import (
            storage_binding,
        )
        from merlin.runtime.engine import (
            storage_binding as captured_constants,
        )
        from merlin.runtime.engine import (
            storage_binding as prepack_authority,
        )
        from merlin.runtime.harness import (
            call_timing as complete_call_timing,
        )
        from merlin.runtime.harness import (
            call_timing as kernel_call_timing,
        )
        from merlin.runtime.harness import (
            direct_kernel as direct_kernel_harness,
        )
        from merlin.runtime.harness import (
            pointer_storage as harness_render,
        )

        # These files decide the emitted harness bytes. The counter's selected
        # Rocket/ISA source pins are verified by the timing selections and
        # carried in contract data; they are not renderer sources.
        self.complete_call_timing()
        kernel_call_timing.from_contract(self._data()["contract"])
        return (
            Path(complete_call_timing.__file__).resolve(),
            Path(direct_kernel_harness.__file__).resolve(),
            Path(harness_render.__file__).resolve(),
            Path(kernel_call_timing.__file__).resolve(),
            *(
                Path(module.__file__).resolve()
                for module in (storage_binding, prepack_authority, captured_constants, storage_encoding, tensor_storage)
            ),
        )

    def _engine(self, engine):
        selection = self._block()
        block = selection["engines"].get(engine)
        _require(block is not None, "requested execution engine is not selected")
        _validate_engine(
            engine,
            block,
            timeout_ceiling=_execution_timeout_ceiling(selection),
            spike_timeout_ceiling=_model_spike_timeout_ceiling(selection),
        )
        return block

    def execution_timeout_limit(self, simulator):
        """The immutable selected engine's process deadline ceiling."""
        return self._engine(simulator)["max_timeout_s"]

    def execution_argv_template(self, simulator):
        """Original command, optionally preloading this same verified ELF before hart release."""
        argv = list(self._engine(simulator)["argv"])
        if simulator == "gsim" and _gsim_startup_transport() is not None:
            _require(
                not any(arg.startswith("+loadmem=") for arg in argv),
                "selected gSIM startup preload conflicts with another ELF selection",
            )
            # The original registered native harness writes only declared DRAM
            # during load_program(). Later TSI/coherent reads remain live RTL.
            argv.append("+loadmem={elf}")
        return tuple(argv)

    def _engine_identity(self, engine):
        """The selected engine's validated identity, reused while its pinned files are unchanged."""
        from merlin.runtime.backends.rocc import tool_selection as M

        block = self._engine(engine)
        paths = [block["binary"]["path"]]
        if isinstance(block.get("receipt"), dict):
            paths += M.receipt_paths(block["receipt"]["path"])
        key = (
            self.target,
            engine,
            hashlib.sha256(self._selection_json.encode()).hexdigest(),
            json.dumps(block, sort_keys=True, default=str),
            os.environ.get("MERLIN_GSIM_STARTUP_PRELOAD", ""),
        )
        return M.cached(key, paths, lambda: self._validated_engine_identity(engine))

    def _validated_engine_identity(self, engine):
        block = self._engine(engine)
        binary = _check_pin(block["binary"], executable=True)
        identity = {
            "engine": engine,
            "binary": block["binary"],
            "selection_sha256": hashlib.sha256(self._selection_json.encode()).hexdigest(),
        }
        if "readback_transport" in block:
            identity["readback_transport"] = block["readback_transport"]
        if "execution_resources" in self._block():
            identity["execution_resources"] = {
                **self._block()["execution_resources"],
                "selected_engine_timeout_s": block["max_timeout_s"],
            }
        if engine == "gsim":
            from merlin.target.simulators import gsim_emulator as GE

            receipt = _check_pin(block["receipt"])
            status, _, lineage = GE._validate_receipt(self.target, binary, block["binary"]["sha256"], receipt)
            _require(
                status == "bound"
                and lineage["schema_version"] == GE.STRICT_RECEIPT_SCHEMA
                and lineage["firrtl_sha256"] == self._execution_firrtl_sha256(),
                "selected engine has no strict receipt for the original FIRRTL",
            )
            # The existing strict checker checks pin bytes, but the actual binary operand
            # must also equal the receipt's artifact path, not an equal-byte sibling.
            doc = json.loads(receipt.read_text(encoding="utf-8"))
            _require(
                doc["artifacts"]["binary"]["path"] == str(binary),
                "selected engine receipt identifies another executable",
            )
            identity["receipt"] = lineage
            startup = _gsim_startup_transport()
            if startup is not None:
                self.execution_argv_template(engine)  # Refuse conflicting selected startup operands.
                identity["startup_transport"] = startup
        elif engine == "verilator":
            receipt = _check_pin(block["receipt"])
            try:
                identity["receipt"] = validate(
                    target=self.target,
                    config=self._block()["config"],
                    original_fir_sha256=self._execution_firrtl_sha256(),
                    binary=binary,
                    receipt=receipt,
                    checked_pin=_check_pin,
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise RoCCExecutionError("selected elaborated RTL build identity is unavailable or changed") from exc
        else:
            from merlin.target.registry import identity as target_registry
            from merlin.target.simulators import spike_runtime as spike_extension

            contract = self._data()["contract"]
            selected = contract["runner"]["spike_extension"]
            library = _check_pin({"path": selected["extlib"], "sha256": selected["sha256"]})
            with target_registry.observed_contract(self.target, contract):
                extension = spike_extension.resolve(
                    self.target, default_library_dir=library.parent, default_extension_name=selected["extension_name"]
                )
            _require(
                extension.declared and extension.extlib == library, "functional extension selection is unavailable"
            )
            identity["extension"] = {
                "path": str(library),
                "sha256": selected["sha256"],
                "flags": list(extension.spike_flags()),
            }
        return identity

    def execution_engine_identity(self, engine):
        """Reopen the selected engine's build identity; execution gates remain separate."""
        return self._engine_identity(engine)

    def _status(self, engine):
        try:
            self._engine_identity(engine)
            return True, "selected prebuilt execution inputs are byte-bound; runtime and timing unqualified"
        except (OSError, ValueError, KeyError, TypeError):
            return False, "selected prebuilt execution inputs are unavailable or changed"

    def available(self, simulator=None):
        if simulator is not None:
            return self._status(simulator)[0]
        return any(self._status(engine)[0] for engine in self._block()["engines"])

    def gsim_status(self):
        return self._status("gsim")

    def spike_status(self):
        return self._status("spike")

    def verilator_status(self):
        return self._status("verilator")

    def gsim_selected_firrtl_status(self):
        return self.gsim_status()

    def gsim_resolution(self):
        from merlin.target.simulators.gsim_emulator import Resolution

        identity = self._engine_identity("gsim")
        return Resolution(
            target=self.target,
            path=_path(identity["binary"]["path"]),
            source="contract",
            ok=True,
            reason="selected prebuilt engine has strict file and transcript identity; runtime and timing unqualified",
            receipt=identity["receipt"],
            receipt_status="bound",
            digest=identity["binary"]["sha256"],
            flavour="binary",
        )

    def gsim_path(self):
        self._engine_identity("gsim")
        return _path(self._engine("gsim")["binary"]["path"])

    def spike_path(self):
        self._engine_identity("spike")
        return _path(self._engine("spike")["binary"]["path"])

    def spike_extension(self):
        """Expose the exact verified functional extension used by this runner."""
        extension = self._engine_identity("spike")["extension"]
        return tuple(extension["flags"]), _path(extension["path"]).parent

    def verilator_path(self):
        self._engine_identity("verilator")
        return _path(self._engine("verilator")["binary"]["path"])

    _gsim_binary = gsim_path
    _spike_binary = spike_path
    _verilator_binary = verilator_path

    def parse_output(self, console, *, readback_policy=None):
        from merlin.runtime.backends.registry import parse_console

        limit = max(row["max_console_bytes"] for row in self._block()["engines"].values())
        data = console.encode("utf-8") if type(console) is str else console
        _require(type(data) is bytes and len(data) <= limit, "execution console exceeds its selected byte budget")
        text = data.decode("utf-8")
        # Validate uniqueness/completion around the existing value codec; do not
        # replace it with a target parser or accept overwritten output/metric rows.
        names, metrics, done = set(), set(), False
        for line in text.splitlines():
            parts = line.split()
            if not parts:
                continue
            marker = parts[0]
            _require(marker != "OUTSUM", "selected execution requires complete output values")
            if marker in {"OUT", "OUT_B64_BEGIN", "METRIC", "DONE"} or marker.startswith("OUT_B64_"):
                _require(not done, "selected output frame continues after completion")
            if marker in {"OUT", "OUT_B64_BEGIN"}:
                index = 1 if marker == "OUT" else 2
                _require(
                    len(parts) > index and parts[index] not in names, "selected output frame repeats or omits a name"
                )
                names.add(parts[index])
            elif marker == "METRIC":
                _require(len(parts) == 3 and parts[1] not in metrics, "selected output frame repeats or omits a metric")
                metrics.add(parts[1])
            elif marker == "DONE":
                _require(len(parts) == 1, "selected output completion is malformed")
                done = True
        outputs, raw = parse_console(text, error_cls=RoCCExecutionError)
        selected = self.readback_policy() if readback_policy is None else readback_policy
        _require(
            type(selected) is ReadbackPolicy
            and selected
            in {self.readback_policy(), *(self.engine_readback_policy(e) for e in self._block()["engines"])},
            "selected output parser has no declared full-value policy",
        )
        if selected.transport == FULL_VALUES_B64:
            _require(bool(outputs), "selected execution omitted complete output values")
        return outputs, raw

    def run_declared_model_elf(self, elf, *, timeout, capture_bytes=False):
        """Run an explicitly selected saved-model image with its own machine declarations.

        Ordinary kernel invocations retain their contract argv. Model callers
        separately verify the declared board/ISA and original-source build.
        """
        return self.run_elf(
            elf, simulator="spike", timeout=timeout, capture_bytes=capture_bytes, declared_model_machine=True
        )

    def run_elf(
        self,
        elf,
        *,
        simulator,
        timeout,
        capture_bytes=False,
        declared_model_machine=False,
        memory_readback=None,
        private_max_cycles=None,
        **kwargs,
    ):
        _require(
            not kwargs and type(capture_bytes) is bool and type(declared_model_machine) is bool,
            "selected execution received unsupported options",
        )
        _require(
            private_max_cycles is None
            or (
                simulator == "gsim"
                and memory_readback is None
                and type(private_max_cycles) is int
                and 0 < private_max_cycles <= (1 << 63) - 1
            ),
            "private gSIM cycle budget must be a positive declared integer",
        )
        _require(
            not declared_model_machine or simulator == "spike",
            "saved-model machine declarations select only the functional simulator",
        )
        block = self._engine(simulator)
        _require(
            type(timeout) in (int, float) and math.isfinite(timeout) and 0 < timeout <= block["max_timeout_s"],
            "selected execution requires a bounded deadline",
        )
        from merlin.common.process.ownership import ExecutionDeadline

        process_deadline = ExecutionDeadline.start(timeout)
        before = self._engine_identity(simulator)
        path = Path(elf)
        elf_sha = _digest(path)
        export = None
        if memory_readback is not None:
            from merlin.runtime.backends.rocc import memory_readback as _chipyard_memory_readback
            from merlin.runtime.views.native_simulator import active_native_runtime

            _require(
                active_native_runtime() is None,
                "selected native filesystem view has no approved coherent-export mapping; "
                "select the direct execution mode explicitly or retain the original console transport",
            )
            _require(
                simulator == "gsim"
                and not capture_bytes
                and not declared_model_machine
                and self.engine_readback_policy(simulator).transport
                in {COHERENT_DUMP_V1, COHERENT_PACKET_V1, COHERENT_PACKET_V2},
                "selected engine has no declared coherent full-value transport",
            )
            export = _chipyard_memory_readback.prepare_memory_readback(path, simulator, memory_readback)
            _require(export.elf_sha256 == elf_sha, "selected memory export differs from the original ELF")
            _require(
                memory_readback.get("transport") == self.memory_readback_transport(simulator),
                "selected memory export differs from the declared engine transport",
            )
        cwd = _path(block["cwd"])
        _require(
            cwd.resolve() == cwd and not any(p.is_symlink() for p in (cwd, *cwd.parents)) and cwd.is_dir(),
            "selected execution working directory is unavailable",
        )
        flags = before.get("extension", {}).get("flags", [])
        template = self.execution_argv_template(simulator)
        budget_cycles = _gsim_budget_max_cycles() if simulator == "gsim" and private_max_cycles is None else None
        if private_max_cycles is not None or budget_cycles is not None:
            cycle_args = [arg for arg in template if arg.startswith("+max-cycles=")]
            _require(
                len(cycle_args) == 1 and cycle_args[0].removeprefix("+max-cycles=").isdigit(),
                "selected gSIM command has no unique source-bound cycle limit",
            )
            cycles = private_max_cycles
            if cycles is None:
                cycles = max(int(cycle_args[0].removeprefix("+max-cycles=")), budget_cycles)
            template = tuple(f"+max-cycles={cycles}" if arg == cycle_args[0] else arg for arg in template)
        _require(
            (_gsim_startup_transport() if simulator == "gsim" else None) == before.get("startup_transport"),
            "selected startup transport changed before execution",
        )
        argv = [block["binary"]["path"], *flags, *(arg.replace("{elf}", str(path)) for arg in template)]
        if export is not None:
            _require(
                not any(arg.startswith(("+dump-", "+signature")) for arg in argv),
                "selected engine repeats memory-export arguments",
            )
            argv.extend(export.argv_suffix)
        dependencies = [Path(__file__).resolve(), _path(block["binary"]["path"])]
        process_inputs = [path]
        if export is not None:
            dependencies.append(Path(_chipyard_memory_readback.__file__).resolve())
            process_inputs.append(export.regions_path)
        if declared_model_machine:
            from merlin.compiler.lower.model import toolchain
            from merlin.runtime.backends.baremetal import image_link as spike_execution
            from merlin.runtime.backends.baremetal import image_link as spike_model
            from merlin.runtime.backends.baremetal import model_build as _model_elf_metadata

            selected_machine_tools = (toolchain.nm(), toolchain.readelf())
            machine_tools = tuple(tool.resolve(strict=True) for tool in selected_machine_tools)
            machine_tool_pins = {str(tool): _digest(tool) for tool in machine_tools}
            memory, isa, harts = (
                _model_elf_metadata.declared_memory(path),
                spike_execution.declared_isa(path),
                _model_elf_metadata.declared_harts(path),
            )
            _require(
                memory is not None
                and isa is not None
                and memory[0] >= 0
                and memory[1] > 0
                and memory[0] + memory[1] <= 1 << 64,
                "saved model has no bounded explicit memory/ISA declaration",
            )
            _require(
                harts is None or type(harts) is int and 0 < harts <= 256,
                "saved model has an invalid declared hart count",
            )
            _require(_digest(path) == elf_sha, "saved model changed while reading machine declarations")
            _require(
                tuple(tool.resolve(strict=True) for tool in selected_machine_tools) == machine_tools
                and all(_digest(tool) == machine_tool_pins[str(tool)] for tool in machine_tools),
                "saved-model machine inspection tools changed",
            )
            selected = [arg for arg in argv[1:] if not arg.startswith(("--isa=", "-m", "-p"))]
            machine = [f"--isa={isa}", f"-m{hex(memory[0])}:{hex(memory[1])}"]
            if harts is not None:
                machine.append(f"-p{harts}")
            argv = [argv[0], *machine, *selected]
            dependencies.extend([Path(spike_model.__file__).resolve(), *machine_tools])
        if simulator in {"gsim", "verilator"}:
            dependencies.append(_path(block["receipt"]["path"]))
        else:
            dependencies.append(Path(before["extension"]["path"]))
        from merlin.runtime.views.native_simulator import (
            active_native_runtime,
            forget_native_observation,
            run_in_native_view,
        )

        forget_native_observation()
        if simulator == "gsim" and export is None and active_native_runtime() is not None:
            from merlin.target.simulators.engine_policy import gsim_runtime_slot

            with gsim_runtime_slot(wait_timeout_s=timeout):
                selected_result = run_in_native_view(
                    self,
                    path,
                    argv=argv,
                    environment=block["environment"],
                    cwd=cwd,
                    timeout=process_deadline.remaining("selected execution budget expired while waiting for a slot"),
                    capture_bytes=capture_bytes,
                    unlimited_stack=False,
                    raw_result=True,
                )
            if selected_result is not None:
                selected_result.check_returncode()
                _require(
                    self._engine_identity(simulator) == before and _digest(path) == elf_sha,
                    "selected engine or ELF changed during execution",
                )
                _require(
                    not any(
                        marker.encode() in selected_result.stdout + selected_result.stderr
                        for marker in block["failure_markers"]
                    ),
                    "selected execution reported a declared failure",
                )
                return selected_result.stdout if capture_bytes else selected_result.stdout.decode("utf-8")
        # Records accompany this caller's original ELF, never a discovered runtime.
        with I.observe(
            path.parent,
            stage="selected_rocc_process",
            argv=argv,
            cwd=cwd,
            env=block["environment"],
            inputs=tuple(process_inputs),
            # The manifest is an input above. Only the dump is produced by
            # this actual simulator invocation; completion seals its bytes.
            outputs=(export.output_path,) if export is not None else (),
            dependencies=dependencies,
        ) as observed:
            from merlin.target.simulators.engine_policy import simulation_runtime_slot

            slot = simulation_runtime_slot(simulator, wait_timeout_s=timeout)
            with slot:
                if export is not None:
                    export.revalidate()
                result = _bounded_process(
                    argv,
                    cwd=cwd,
                    env=block["environment"],
                    timeout=process_deadline.remaining("selected execution budget expired while waiting for a slot"),
                    max_bytes=block["max_console_bytes"],
                    observed=observed,
                )
            result.check_returncode()
            _require(
                self._engine_identity(simulator) == before and _digest(path) == elf_sha,
                "selected engine or ELF changed during execution",
            )
            if export is not None:
                export.revalidate(completed=True)
            I.verify(observed.path)
            I.require_environment(observed.path, environment=block["environment"])
        data = result.stdout
        _require(
            not any(marker.encode() in data + result.stderr for marker in block["failure_markers"]),
            "selected execution reported a declared failure",
        )
        return data if capture_bytes else data.decode("utf-8")


# ---- rtl_build_receipt -------------------------------------------------------------------------
# Verify observed RTL build identity; execution and correctness remain separate.

SCHEMA = "merlin.elaborated_rtl_build.v1"
ROLES = (
    "rtl_source_conversion",
    "rtl_cpp_emission",
    "rtl_model_compilation",
    "transport_fesvr_configuration",
    "transport_fesvr_compilation",
    "rtl_native_link",
)


def validate(*, target, config, original_fir_sha256, binary, receipt, checked_pin):
    """Bind selected bytes to joined producer observations, never qualify their execution.

    The caller supplies its ordinary-file/hash checker. No stage is discovered,
    rebuilt or borrowed. Exact source/command correspondence remains the reviewed
    producer's responsibility; observation metadata does not prove RTL equivalence.
    """

    def require(yes, message):
        if not yes:
            raise ValueError(message)

    def mapping(value, keys):
        require(type(value) is dict and set(value) == set(keys), "RTL build receipt has unsupported fields")

    def pin(value):
        mapping(value, ("path", "sha256"))
        checked_pin(value)
        return value

    def roster(value):
        require(type(value) is list and 0 < len(value) <= 4096, "RTL build artifact roster is absent or unbounded")
        require(
            len({r.get("path") for r in value if type(r) is dict}) == len(value),
            "RTL build artifact roster is ambiguous",
        )
        return [pin(r) for r in value]

    def contains(actual, expected):
        return all(p in actual for p in expected)

    require(Path(receipt).stat().st_size <= 4 * 1024 * 1024, "RTL build receipt exceeds its bounded size")
    document = json.loads(Path(receipt).read_bytes())
    mapping(
        document,
        (
            "schema",
            "engine",
            "target",
            "config",
            "firrtl",
            "binary",
            "hdl",
            "model_sources",
            "model_archives",
            "transport_sources",
            "transport_configuration",
            "transport_library",
            "bridge_sources",
            "driver_source",
            "stages",
            "memory_policy",
            "reset_policy",
        ),
    )
    require(
        document["schema"] == SCHEMA and document["engine"] == "verilator",
        "selected RTL engine has another build receipt kind",
    )
    require(
        document["target"] == target and document["config"] == config,
        "selected RTL build identifies another target or configuration",
    )
    source = pin(document["firrtl"])
    artifact = pin(document["binary"])
    require(source["sha256"] == original_fir_sha256, "selected RTL build does not bind the original FIRRTL")
    require(Path(artifact["path"]) == Path(binary), "selected RTL build identifies another executable")
    for name in ("memory_policy", "reset_policy"):
        require(type(document[name]) is dict and document[name], "RTL build omits its explicit platform policy")
    hdl = roster(document["hdl"])
    model = roster(document["model_sources"])
    archives = roster(document["model_archives"])
    transport = roster(document["transport_sources"])
    configuration = roster(document["transport_configuration"])
    library = pin(document["transport_library"])
    bridge = roster(document["bridge_sources"])
    driver = pin(document["driver_source"])
    stages = document["stages"]
    require(type(stages) is list and len(stages) == len(ROLES), "RTL build has no complete observed stage chain")
    observations = {}
    for role, stage in zip(ROLES, stages):
        mapping(stage, ("role", "invocation", "environment"))
        require(stage["role"] == role, "RTL build stages are missing or out of order")
        selected = pin(stage["invocation"])
        require(
            type(stage["environment"]) is dict
            and all(type(k) is str and type(v) is str for k, v in stage["environment"].items()),
            "RTL build has no explicit stage environment",
        )
        observed = I.require_environment(Path(selected["path"]), environment=stage["environment"])
        require(
            observed.get("kind") == "subprocess" and observed.get("stage") == role,
            "RTL build attaches an unrelated successful invocation",
        )
        observations[role] = observed
    convert, emit, compile, configure, tcompile, link = [observations[r] for r in ROLES]
    require(
        contains(convert["inputs"], [source]) and contains(convert["outputs"], hdl),
        "RTL conversion loses its original source or produced HDL",
    )
    require(
        contains(emit["inputs"], hdl) and contains(emit["outputs"], model),
        "RTL emission does not consume the selected HDL and produce the selected model",
    )
    require(
        contains(compile["inputs"], model) and contains(compile["outputs"], archives),
        "RTL compilation does not consume the selected model and produce its archives",
    )
    require(
        contains(configure["inputs"], transport) and contains(configure["outputs"], configuration),
        "transport configuration loses its selected public source or products",
    )
    require(
        contains(tcompile["inputs"], transport + configuration) and contains(tcompile["outputs"], [library]),
        "transport compilation does not produce the selected library",
    )
    require(
        contains(link["inputs"], archives + [library, driver] + bridge) and contains(link["outputs"], [artifact]),
        "native link does not consume the selected model and transport or produce the selected executable",
    )
    return {
        "schema": SCHEMA,
        "engine": "verilator",
        "target": target,
        "config": config,
        "firrtl": source,
        "binary": artifact,
        "receipt_path": str(receipt),
        "scope": (
            "observed source/build byte identity only; "
            "execution, equivalence, correctness and hardware timing remain unqualified"
        ),
    }
