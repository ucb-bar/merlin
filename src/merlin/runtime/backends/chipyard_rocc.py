"""Contract-selected execution of prebuilt RoCC simulators and logical LLVM kernels.

This module contains no target compiler or compute implementation. Runtime units
are a closed set of core console/startup primitives; simulator and software ABI
choices come from the selected contract. File/transcript checks establish local
identity, not emitter correspondence, hardware timing or runtime qualification.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import selectors
import signal
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.common.paths import checkout_root, python_source_dir
from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe, KernelStackFramePolicy
from merlin.targetgen.contract.readback_policy import FULL_VALUES_B64

_RUNTIME = {
    "startup": "spike/crt.S",
    "htif_console": "spike/htif.c",
    "printf": "spike/printf_min.c",
    "libc": "spike/libc_min.c",
}
_HEADERS = {
    "htif": "spike/htif.h",
    "out_b64": "out_b64.h",
    "out_bin": "out_bin.h",
    "out_bin_memory": "out_bin_memory.h",
}
_CFLAGS = frozenset(
    {
        "-O0",
        "-O1",
        "-O2",
        "-O3",
        "-Os",
        "-Og",
        "-g",
        "-ffreestanding",
        "-nostdlib",
        "-nostartfiles",
        "-fno-builtin",
        "-fno-tree-vectorize",
        "-fno-vectorize",
        "-fno-slp-vectorize",
        "-fno-tree-loop-distribute-patterns",
        "-ffunction-sections",
        "-fdata-sections",
        "-fno-pie",
        "-fno-pic",
        "-fno-stack-protector",
        "-mno-relax",
        "-std=c11",
        "-std=gnu11",
    }
)
_LDFLAGS = frozenset({"-Wl,--gc-sections", "-Wl,--no-relax", "-Wl,--build-id=none", "-no-pie"})
_READOUT = {"scalar_abi", "epilogue_capability", "operand_sum", "stage_routes"}
_SCOPE = "selected process and complete console only; hardware, timing and runtime qualification unestablished"


class RoCCExecutionError(ValueError):
    """A selected execution input is missing, unsupported or changed."""


def _require(condition, message):
    if not condition:
        raise RoCCExecutionError(message)


def _fields(value, required, optional=()):
    _require(
        type(value) is dict and set(required) <= set(value) <= set(required) | set(optional),
        "execution selection has unsupported or missing fields",
    )


def _positive(value):
    _require(type(value) is int and 0 < value < 2**63, "execution budget requires a positive bounded integer")


def _path(value):
    _require(
        type(value) is str and "\0" not in value and Path(value).is_absolute() and ".." not in Path(value).parts,
        "execution selection requires an explicit absolute path",
    )
    return Path(value)


def _sha(value):
    _require(
        type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
        "execution selection requires an exact SHA256 digest",
    )


def _pin_shape(value):
    _fields(value, {"path", "sha256"})
    _path(value["path"])
    _sha(value["sha256"])


def _regular(path):
    try:
        okay = (
            not any(p.is_symlink() for p in (path, *path.parents))
            and path.resolve() == path
            and stat.S_ISREG(path.stat().st_mode)
        )
    except OSError:
        okay = False
    _require(okay, "selected execution member is not a canonical regular file")


def _digest(path):
    _regular(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _check_pin(pin, *, executable=False):
    path = _path(pin["path"])
    _require(_digest(path) == pin["sha256"], "selected execution member changed")
    _require(not executable or os.access(path, os.X_OK), "selected execution tool is not executable")
    return path


def _resource(identifier, roster):
    # Do not honor runtime/repo overrides that could turn an ID into vendor code.
    owner = checkout_root()
    root = owner / "merlin/runtime" if owner is not None else python_source_dir() / "merlin/_data/runtime"
    return root / "baremetal" / roster[identifier]


def _units(rows, roster):
    _require(type(rows) is list and rows, "execution recipe requires explicit core runtime selections")
    names = []
    for row in rows:
        _fields(row, {"id", "sha256"})
        _require(
            type(row["id"]) is str and row["id"] in roster, "execution recipe includes an unsupported runtime unit"
        )
        _sha(row["sha256"])
        names.append(row["id"])
    _require(len(set(names)) == len(names), "execution recipe repeats a runtime unit")


def _strings(value):
    _require(
        type(value) is list and all(type(x) is str and x and "\0" not in x for x in value),
        "execution selection requires exact string arguments",
    )


def _validate_toolchain(block, abi):
    _fields(
        block,
        {
            "compiler",
            "runtime_units",
            "headers",
            "link_script",
            "load_address",
            "cflags",
            "ldflags",
            "kernel_stack_frame",
        },
    )
    _pin_shape(block["compiler"])
    _pin_shape(block["link_script"])
    _positive(block["load_address"])
    _units(block["runtime_units"], _RUNTIME)
    _units(block["headers"], _HEADERS)
    _require(
        {"startup", "htif_console"} <= {r["id"] for r in block["runtime_units"]}
        and "htif" in {r["id"] for r in block["headers"]},
        "execution recipe omits the selected startup or console ABI",
    )
    required_headers = (
        {"htif", "out_b64"}
        if abi.readback_policy.transport == FULL_VALUES_B64
        else {"htif", "out_bin", "out_bin_memory"}
    )
    _require(
        required_headers <= {row["id"] for row in block["headers"]},
        "execution recipe omits its complete selected readback header roster",
    )
    _strings(block["cflags"])
    _strings(block["ldflags"])
    flags = block["cflags"]
    for prefix in ("-march=", "-mabi="):
        _require(
            sum(flag.startswith(prefix) and len(flag) > len(prefix) for flag in flags) == 1,
            "execution recipe requires one explicit ISA and ABI selection",
        )
    for flag in flags:
        _require(
            flag in _CFLAGS
            or any(
                flag.startswith(prefix)
                and flag[len(prefix) :].isascii()
                and all(c.isalnum() or c in "_-." for c in flag[len(prefix) :])
                and flag[len(prefix) :]
                for prefix in ("-march=", "-mabi=", "-mcmodel=")
            ),
            "execution recipe contains an unsupported compiler option",
        )
    _require(
        all(flag in _LDFLAGS for flag in block["ldflags"]), "execution recipe contains an unsupported linker option"
    )
    stack = block["kernel_stack_frame"]
    _fields(stack, {"entry_symbol", "max_static_bytes"})
    _require(stack["entry_symbol"] == abi.pointer_abi.entry_symbol, "execution recipe and harness entry differ")
    _positive(stack["max_static_bytes"])


def _validate_engine(engine, block):
    required = {"binary", "argv", "environment", "cwd", "max_timeout_s", "max_console_bytes", "failure_markers"}
    _fields(block, required | ({"receipt"} if engine == "gsim" else set()))
    _pin_shape(block["binary"])
    if engine == "gsim":
        _pin_shape(block["receipt"])
    _path(block["cwd"])
    _strings(block["argv"])
    _require(
        block["argv"].count("{elf}") == 1
        and all(token == "{elf}" or "{" not in token and "}" not in token for token in block["argv"]),
        "execution command requires exactly one ELF operand",
    )
    _require(
        type(block["environment"]) is dict
        and all(
            type(k) is str and k and "=" not in k and "\0" not in k and type(v) is str and "\0" not in v
            for k, v in block["environment"].items()
        ),
        "execution requires an explicit process environment",
    )
    _positive(block["max_timeout_s"])
    _require(block["max_timeout_s"] <= 600, "execution deadline exceeds the bounded process limit")
    _positive(block["max_console_bytes"])
    _strings(block["failure_markers"])
    if engine == "spike":
        _require(
            any(token.startswith("--isa=") and len(token) > 6 for token in block["argv"])
            and not any(token.startswith(("--extension", "--extlib")) for token in block["argv"]),
            "functional execution requires an explicit ISA and a single selected extension",
        )


def _validate_readout(block):
    _fields(block, (), _READOUT)
    for key in ("scalar_abi", "operand_sum"):
        if key in block:
            _require(block[key] is None or type(block[key]) is dict, "readout observation is not a data mapping")
    for key in ("epilogue_capability", "stage_routes"):
        if key in block:
            _require(
                block[key] is None or type(block[key]) is list and all(type(r) is dict for r in block[key]),
                "readout observation is not a complete data roster",
            )


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

    def _data(self):
        return json.loads(self._selection_json)

    def _block(self):
        return self._data()["contract"]["runner"]["chipyard_rocc"]

    @property
    def ORACLE(self):
        return {
            engine: {
                "kind": "functional" if engine == "spike" else "rtl",
                "fidelity": "functional_model" if engine == "spike" else "elaborated_rtl",
                "derived_from_rtl": engine == "gsim",
                "scope": _SCOPE,
            }
            for engine in self._block()["engines"]
        }

    def _readout(self, key):
        return self._block().get("readout", {}).get(key)

    def readback_policy(self):
        from merlin.runtime.harness_render import validate_contract

        return validate_contract(self._data()["contract"]).readback_policy

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

    def render_harness(self, cb, *, target, inputs=None, readback_policy=None, warm_profile=None):
        _require(target == self.target, "logical harness target differs from the selected backend")
        if warm_profile is not None:
            raise NotImplementedError("selected logical harness has no qualified warm profile transport")
        from merlin.runtime.harness_render import render_harness

        return render_harness(
            cb, target=target, inputs=inputs, contract=self._data()["contract"], readback_policy=readback_policy
        )

    def _engine(self, engine):
        block = self._block()["engines"].get(engine)
        _require(block is not None, "requested execution engine is not selected")
        return block

    def _engine_identity(self, engine):
        block = self._engine(engine)
        binary = _check_pin(block["binary"], executable=True)
        identity = {
            "engine": engine,
            "binary": block["binary"],
            "selection_sha256": hashlib.sha256(self._selection_json.encode()).hexdigest(),
        }
        if engine == "gsim":
            from merlin.targetgen import gsim_emulator as GE

            receipt = _check_pin(block["receipt"])
            status, _, lineage = GE._validate_receipt(self.target, binary, block["binary"]["sha256"], receipt)
            _require(
                status == "bound"
                and lineage["schema_version"] == GE.STRICT_RECEIPT_SCHEMA
                and lineage["firrtl_sha256"] == self._data()["facts"]["inputs"]["fir_sha256"],
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
        else:
            from merlin.targetgen import spike_extension, target_registry

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
        from merlin.targetgen.gsim_emulator import Resolution

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

    def verilator_path(self):
        self._engine("verilator")

    _gsim_binary = gsim_path
    _spike_binary = spike_path
    _verilator_binary = verilator_path

    def parse_output(self, console):
        from merlin.runtime.backends.base import parse_console

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
        if self._data()["contract"]["harness_abi"]["readback_transport"] == FULL_VALUES_B64:
            _require(bool(outputs), "selected execution omitted complete output values")
        return outputs, raw

    def run_elf(self, elf, *, simulator, timeout, capture_bytes=False, **kwargs):
        _require(not kwargs and type(capture_bytes) is bool, "selected execution received unsupported options")
        block = self._engine(simulator)
        _require(
            type(timeout) in (int, float) and math.isfinite(timeout) and 0 < timeout <= block["max_timeout_s"],
            "selected execution requires a bounded deadline",
        )
        before = self._engine_identity(simulator)
        path = Path(elf)
        elf_sha = _digest(path)
        cwd = _path(block["cwd"])
        _require(
            cwd.resolve() == cwd and not any(p.is_symlink() for p in (cwd, *cwd.parents)) and cwd.is_dir(),
            "selected execution working directory is unavailable",
        )
        flags = before.get("extension", {}).get("flags", [])
        argv = [block["binary"]["path"], *flags, *(str(path) if arg == "{elf}" else arg for arg in block["argv"])]
        dependencies = [Path(__file__).resolve(), _path(block["binary"]["path"])]
        if simulator == "gsim":
            dependencies.append(_path(block["receipt"]["path"]))
        else:
            dependencies.append(Path(before["extension"]["path"]))
        # Records accompany this caller's original ELF, never a discovered runtime.
        with I.observe(
            path.parent,
            stage="selected_rocc_process",
            argv=argv,
            cwd=cwd,
            env=block["environment"],
            inputs=(path,),
            dependencies=dependencies,
        ) as observed:
            result = _bounded_process(
                argv,
                cwd=cwd,
                env=block["environment"],
                timeout=timeout,
                max_bytes=block["max_console_bytes"],
                observed=observed,
            )
            result.check_returncode()
            _require(
                self._engine_identity(simulator) == before and _digest(path) == elf_sha,
                "selected engine or ELF changed during execution",
            )
            I.verify(observed.path)
            I.require_environment(observed.path, environment=block["environment"])
        data = result.stdout
        _require(
            not any(marker.encode() in data for marker in block["failure_markers"]),
            "selected execution reported a declared failure",
        )
        return data if capture_bytes else data.decode("utf-8")


def _bounded_process(argv, *, cwd, env, timeout, max_bytes, observed):
    """Capture one selected process group, retaining finite partial output on refusal."""
    process = subprocess.Popen(
        argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True
    )
    data = bytearray()
    deadline = time.monotonic() + timeout
    try:
        with selectors.DefaultSelector() as poll:
            poll.register(process.stdout, selectors.EVENT_READ)
            while poll.get_map():
                left = deadline - time.monotonic()
                if left <= 0:
                    raise subprocess.TimeoutExpired(argv, timeout, output=bytes(data))
                for key, _ in poll.select(min(left, 0.1)):
                    chunk = os.read(key.fileobj.fileno(), min(65536, max_bytes + 1 - len(data)))
                    if not chunk:
                        poll.unregister(key.fileobj)
                    else:
                        data.extend(chunk)
                        _require(len(data) <= max_bytes, "execution console exceeds its selected byte budget")
            process.wait(timeout=max(0.001, deadline - time.monotonic()))
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        observed.complete(subprocess.CompletedProcess(argv, process.returncode, stdout=bytes(data), stderr=b""))
        raise
    finally:
        process.stdout.close()
    result = subprocess.CompletedProcess(argv, process.returncode, stdout=bytes(data), stderr=b"")
    observed.complete(result)
    return result


def bind(*, target, contract, facts):
    """Bind already selected data once; no plugin, discovery, binary check or build."""
    _require(
        type(target) is str and target and type(contract) is dict and type(facts) is dict and facts,
        "RoCC execution requires an explicit contract and original facts",
    )
    try:
        snapshot = json.dumps(
            {"contract": contract, "facts": facts}, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise RoCCExecutionError("execution selection is not finite serializable data") from exc
    selected = json.loads(snapshot)
    contract, facts = selected["contract"], selected["facts"]
    runner = contract.get("runner")
    _require(type(runner) is dict and runner.get("backend") == "chipyard_rocc", "RoCC execution family is not selected")
    block = runner.get("chipyard_rocc")
    _fields(block, {"version", "config", "toolchain", "engines"}, {"readout"})
    _require(
        type(block["version"]) is int and block["version"] == 1 and type(block["config"]) is str and block["config"],
        "RoCC execution requires a supported version and explicit configuration",
    )
    inputs = facts.get("inputs")
    body = facts.get("facts")
    consistency = facts.get("source_consistency")
    _require(
        type(inputs) is dict and type(body) is dict and type(body.get("source")) is dict and type(consistency) is dict,
        "selected original facts omit their source configuration binding",
    )
    configs = [body["source"].get("config"), consistency.get("config")]
    if "source" in facts:
        _require(type(facts["source"]) is dict, "selected original facts have an ambiguous source configuration")
        configs.append(facts["source"].get("config"))
    _require(
        contract.get("name") == target
        and inputs.get("target") == target
        and all(type(config) is str and config == block["config"] for config in configs),
        "selected contract or original facts identify another target or configuration",
    )
    _sha(inputs.get("fir_sha256"))
    from merlin.runtime.harness_render import validate_contract
    from merlin.targetgen.rocc.semantics import bind as bind_semantics

    abi = validate_contract(contract)
    _require(abi.readback_policy.transport == FULL_VALUES_B64, "selected execution has no coherent readback transport")
    _validate_toolchain(block["toolchain"], abi)
    _require(
        type(block["engines"]) is dict and block["engines"] and set(block["engines"]) <= {"spike", "gsim"},
        "RoCC execution requires explicitly supported engines",
    )
    for engine, engine_block in block["engines"].items():
        _validate_engine(engine, engine_block)
    if "spike" in block["engines"]:
        extension = runner.get("spike_extension")
        _fields(extension, {"extension_name", "extlib", "sha256"})
        _require(
            type(extension["extension_name"]) is str
            and extension["extension_name"]
            and all(c.isalnum() or c == "_" for c in extension["extension_name"]),
            "functional execution requires a plain selected extension name",
        )
        _path(extension["extlib"])
        _sha(extension["sha256"])
    if "readout" in block:
        _validate_readout(block["readout"])
    semantics = bind_semantics(target=target, contract=contract, facts=facts)
    return BoundRoCCBackend(target, snapshot, semantics)
