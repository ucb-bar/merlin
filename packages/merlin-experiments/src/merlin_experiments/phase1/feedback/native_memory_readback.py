"""Trusted, invocation-only memory readback hook for a selected native backend.

This prepares exact output storage before a simulator can inspect the ELF. Core
still owns execution, normal exit, DONE, build receipts and numerical grading.
"""

from __future__ import annotations

import hashlib
import json
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from merlin.common.jsonio import canonical_json

from .native_output_readback import (
    admit_coherent_output_layout,
    admit_fixed_address_output_layout,
    admit_htif_signature_bounds,
    decode_coherent_output_dump,
    decode_htif_signature,
)

_TRANSPORTS = {
    ("spike", "htif_signature_v1"),
    ("gsim", "gsim_coherent_dump_v1"),
    ("gsim", "gsim_coherent_packet_v1"),
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _same_selected_backend(backend, selected, simulator):
    """Modules retain identity; exact bound values must also reopen engine pins."""
    from merlin.runtime.backends.chipyard_rocc_bound import BoundRoCCBackend

    if type(backend) is BoundRoCCBackend or type(selected) is BoundRoCCBackend:
        return (
            type(backend) is type(selected) is BoundRoCCBackend
            and backend.target == selected.target
            and backend._selection_json == selected._selection_json
            and backend.execution_engine_identity(simulator) == selected.execution_engine_identity(simulator)
        )
    return backend is selected


def _archive_if_present(path: Path, value: object) -> None:
    """Preserve only actual subprocess bytes without inventing a completion."""
    if type(value) not in (str, bytes):
        return
    raw = value if type(value) is bytes else value.encode("utf-8")
    with path.open("xb") as output:
        output.write(raw)


def select_memory_engine(
    *, target: str, simulator: str, facts_path: Path, backend: Any
) -> tuple[dict, Callable[[], dict]]:
    """Bind the actual public memory runner to explicit facts and engine bytes.

    The callback returns the same citation only after reselecting the engine and
    rechecking the facts/FIRRTL and engine bytes. It is required before launch,
    after execution, and after decoding; no facts cache may be regenerated here.
    """
    from merlin.compile.model_execution_inputs import native_engine, selected_firrtl
    from merlin.runtime.backends import base as backends
    from merlin.targetgen.oracle_policy import selected_l3_engine_report

    facts_path = Path(facts_path)

    def selected_facts() -> dict:
        if facts_path.is_symlink() or not facts_path.is_file():
            raise ValueError("memory engine has no ordinary selected RTL facts")
        document = json.loads(facts_path.read_text(encoding="utf-8"))
        if type(document) is not dict or type(document.get("facts")) is not dict:
            raise ValueError("memory engine has no selected RTL source record")
        source = document["facts"].get("source")
        config = source.get("config") if type(source) is dict else None
        if type(config) is not str or not config:
            raise ValueError("memory engine has no explicit RTL source config")
        return selected_firrtl(facts_path, target=target, config=config)

    def resolve() -> tuple[dict, Callable[[], None]]:
        if type(target) is not str or not target or not _same_selected_backend(
            backend, backends.get_backend(target), simulator
        ):
            raise ValueError("memory runner differs from the selected target backend")
        facts = selected_facts()
        if simulator == "spike":
            from merlin.targetgen.native_model_execution import _functional_engine

            actual, engine, revalidate_engine = _functional_engine(target)
            policy = None
        elif simulator == "gsim":
            policy = selected_l3_engine_report(target)
            if type(policy) is not dict or policy.get("available") is not True or policy.get("engine") != "gsim":
                raise ValueError("GSim is not the selected RTL engine")
            actual, engine, revalidate_engine, _ = native_engine(target, "gsim", facts)
        else:
            raise ValueError("memory runner has no reviewed simulator binding")
        if not _same_selected_backend(backend, actual, simulator):
            raise ValueError("memory runner uses a backend outside the selected engine")
        revalidate_engine()
        citation = {
            "schema": "merlin.native_memory_engine.v1",
            "target": target,
            "simulator": simulator,
            "rtl_facts": facts,
            "engine": engine,
            "selected_l3": policy,
        }
        # Canonical JSON also refuses a non-serializable or non-finite citation.
        return json.loads(canonical_json(citation)), revalidate_engine

    initial, _ = resolve()
    initial_raw = canonical_json(initial)

    def revalidate() -> dict:
        current, _ = resolve()
        if canonical_json(current) != initial_raw:
            raise ValueError("selected memory runner or hardware facts changed during execution")
        return json.loads(initial_raw)

    return json.loads(initial_raw), revalidate


class NativeMemoryReadback:
    """One-use trusted hook; no candidate command-buffer field can select it."""

    def __init__(self, *, facts_path: Path, policy=None) -> None:
        from merlin.targetgen.contract.readback_policy import MEMORY_TRANSPORTS, selected

        self.facts_path = Path(facts_path)
        self.policy = selected(policy)
        if self.policy is not None and self.policy.transport not in MEMORY_TRANSPORTS:
            raise ValueError("memory reader requires an explicit memory transport")
        self._prepared: dict[str, Any] | None = None

    def _transport(self, backend: Any, simulator: str) -> str:
        from merlin.targetgen.contract.readback_policy import COHERENT_PACKET_V1

        declared = getattr(backend, "memory_readback_transport", None)
        if not callable(declared):
            raise ValueError("selected backend does not declare a memory readback transport")
        packet = self.policy is not None and self.policy.transport == COHERENT_PACKET_V1
        transport = declared(simulator, policy_transport=self.policy.transport) if packet else declared(simulator)
        expected = (
            "htif_signature_v1"
            if simulator == "spike"
            else ("gsim_coherent_packet_v1" if packet else "gsim_coherent_dump_v1")
        )
        if type(transport) is not str or (simulator, transport) not in _TRANSPORTS or transport != expected:
            raise ValueError("selected backend cannot run the requested memory transport")
        return transport

    def prepare(
        self,
        *,
        cb: dict,
        target: str,
        elf_path: Path,
        workdir: Path,
        simulator: str,
        backend: Any,
    ) -> dict:
        """Return a closed request only after exact source, ELF and range preflight."""
        from merlin.runtime.backends import base as backends

        if self._prepared is not None:
            raise ValueError("memory readback hook cannot prepare twice")
        if type(cb) is not dict or type(target) is not str or not target:
            raise ValueError("memory readback requires a concrete selected command buffer and target")
        cb_raw = canonical_json(cb)
        if not _same_selected_backend(backend, backends.get_backend(target), simulator):
            raise ValueError("memory readback backend differs from selected target provider")
        transport = self._transport(backend, simulator)
        workdir = Path(workdir)
        if not workdir.is_absolute() or workdir.is_symlink() or not workdir.is_dir():
            raise ValueError("memory readback requires an ordinary absolute run-owned workdir")
        # Capsule tiers can share this generated workdir. Each invocation must
        # retain its own source/output bytes without reusing or deleting an
        # earlier tier's evidence.
        source = Path(tempfile.mkdtemp(prefix="memory_readback_", dir=workdir))
        cb_path = source / "command_buffer.json"
        with cb_path.open("xb") as output:
            output.write(cb_raw)
        if cb_path.read_bytes() != cb_raw:
            raise ValueError("staged memory readback command buffer changed")
        elf_path = Path(elf_path)
        if elf_path.is_symlink() or not elf_path.is_file() or not stat.S_ISREG(elf_path.stat().st_mode):
            raise ValueError("memory readback requires an ordinary linked ELF")
        admission = admit_coherent_output_layout(
            submission=source,
            command_buffer_member=cb_path.name,
            target=target,
            facts_path=self.facts_path,
            elf_path=elf_path,
            expected_elf_sha256=_sha(elf_path.read_bytes()),
        )
        fixed = admit_fixed_address_output_layout(
            admission=admission,
            submission=source,
            command_buffer_member=cb_path.name,
            target=target,
            facts_path=self.facts_path,
            elf_path=elf_path,
        )
        bounds = None
        packet_storage = None
        if transport == "htif_signature_v1":
            bounds = admit_htif_signature_bounds(
                admission=admission,
                submission=source,
                command_buffer_member=cb_path.name,
                target=target,
                facts_path=self.facts_path,
                elf_path=elf_path,
            )
        elif transport == "gsim_coherent_packet_v1":
            from .native_packet_readback import admit_coherent_packet_storage

            packet_storage = admit_coherent_packet_storage(
                admission=admission,
                submission=source,
                command_buffer_member=cb_path.name,
                target=target,
                facts_path=self.facts_path,
                elf_path=elf_path,
            )
        regions = (
            [{"base": bounds["begin"], "bytes": bounds["bytes"]}]
            if bounds is not None
            else [
                {"base": row["address"], "bytes": row["bytes"]}
                for row in sorted(admission["outputs"], key=lambda row: row["address"])
            ]
        )
        if packet_storage is not None:
            regions = packet_storage["regions"]
        output_path = source / ("output.signature" if bounds is not None else "output.dump")
        if output_path.exists() or output_path.is_symlink():
            raise ValueError("memory readback output path is not fresh")
        regions_path = None
        regions_raw = None
        if transport in {"gsim_coherent_dump_v1", "gsim_coherent_packet_v1"}:
            regions_path = source / "regions.txt"
            # This is the existing GSim +dump-regions wire format, not a JSON
            # descriptor. The in-memory request below is still closed JSON.
            regions_raw = "".join(f"{row['base']:#x} {row['bytes']}\n" for row in regions).encode("ascii")
            with regions_path.open("xb") as output:
                output.write(regions_raw)
            if regions_path.read_bytes() != regions_raw:
                raise ValueError("memory readback region request changed during staging")
        request = {
            "schema": "oracle_memory_readback_v1",
            "transport": transport,
            "output_path": str(output_path),
            "regions": regions,
            "elf_sha256": admission["elf_sha256"],
            "regions_path": str(regions_path) if regions_path is not None else None,
        }
        self._prepared = {
            "cb": cb,
            "cb_raw": cb_raw,
            "cb_path": cb_path,
            "source": source,
            "target": target,
            "elf_path": elf_path,
            "simulator": simulator,
            "backend": backend,
            "transport": transport,
            "admission": admission,
            "fixed": fixed,
            "bounds": bounds,
            "packet_storage": packet_storage,
            "policy_record": self.policy.record() if self.policy is not None else None,
            "request": request,
            "request_raw": canonical_json(request),
            "output_path": output_path,
            "regions_path": regions_path,
            "regions_raw": regions_raw,
        }
        return {"memory_readback": request}

    def decode(self, console: str | bytes) -> tuple[dict[str, list], dict]:
        """Recheck prelaunch bytes and decode, without asserting exit, DONE or a grade."""
        from merlin.runtime.backends import base as backends

        state = self._prepared
        if state is None:
            raise ValueError("memory readback was not prepared before execution")
        if type(console) not in (str, bytes):
            raise ValueError("memory readback requires the exact archived console type")
        if (
            canonical_json(state["cb"]) != state["cb_raw"]
            or state["cb_path"].is_symlink()
            or state["cb_path"].read_bytes() != state["cb_raw"]
            or state["source"].is_symlink()
            or not state["source"].is_dir()
            or not _same_selected_backend(state["backend"], backends.get_backend(state["target"]), state["simulator"])
            or self._transport(state["backend"], state["simulator"]) != state["transport"]
            or (self.policy.record() if self.policy is not None else None) != state["policy_record"]
            or canonical_json(state["request"]) != state["request_raw"]
        ):
            raise ValueError("memory readback selected request or source changed during execution")
        regions_path = state["regions_path"]
        if regions_path is not None and (
            regions_path.is_symlink()
            or not regions_path.is_file()
            or not stat.S_ISREG(regions_path.stat().st_mode)
            or regions_path.read_bytes() != state["regions_raw"]
        ):
            raise ValueError("memory readback selected region request changed during execution")
        source_kwargs = {
            "admission": state["admission"],
            "submission": state["source"],
            "command_buffer_member": state["cb_path"].name,
            "target": state["target"],
            "facts_path": self.facts_path,
            "elf_path": state["elf_path"],
        }
        fixed = admit_fixed_address_output_layout(**source_kwargs)
        if canonical_json(fixed) != canonical_json(state["fixed"]):
            raise ValueError("memory readback fixed-address preflight changed during execution")
        if state["transport"] == "htif_signature_v1":
            values = decode_htif_signature(
                **source_kwargs,
                bounds_admission=state["bounds"],
                signature_path=state["output_path"],
            )
            artifact_sha256 = values["signature_sha256"]
        elif state["packet_storage"] is not None:
            from .native_packet_readback import decode_coherent_packet_dump

            values = decode_coherent_packet_dump(
                **source_kwargs,
                packet_admission=state["packet_storage"],
                dump_path=state["output_path"],
            )
            artifact_sha256 = values["dump_sha256"]
        else:
            values = decode_coherent_output_dump(
                **source_kwargs,
                fixed_address_admission=fixed,
                dump_path=state["output_path"],
            )
            artifact_sha256 = values["dump_sha256"]
        evidence = {
            "schema": (
                "oracle_memory_readback_evidence_v3"
                if state["packet_storage"] is not None
                else "oracle_memory_readback_evidence_v2"
                if state["bounds"] is not None and state["bounds"]["schema"] == "htif_signature_bounds_v2"
                else "oracle_memory_readback_evidence_v1"
            ),
            "status": "complete",
            "scope": values.get("scope")
            or (
                "complete physical-region and logical-output decoding only; normal exit, DONE, "
                "numerical correctness, placement and certification are caller obligations"
            ),
            "transport": state["transport"],
            "request_sha256": _sha(state["request_raw"]),
            "command_buffer_sha256": _sha(state["cb_raw"]),
            "elf_sha256": state["admission"]["elf_sha256"],
            "layout_admission_sha256": values["layout_admission_sha256"],
            "fixed_address_admission_sha256": _sha(canonical_json(fixed)),
            "bounds_admission_sha256": values.get("bounds_admission_sha256"),
            "artifact_sha256": artifact_sha256,
            "console_sha256": _sha(console if type(console) is bytes else console.encode("utf-8")),
            "source": values["source"],
        }
        if state["packet_storage"] is not None:
            evidence["packet_admission_sha256"] = values["packet_admission_sha256"]
            evidence["payload_bytes"] = values["payload_bytes"]
        return values["outputs"], evidence


class NativeLogicalMemoryReadback:
    """One-use complete dense reader for the original neutral pointer harness.

    The original build receipt binds that harness and command buffer. This
    reader independently admits its actual sized writable tensor objects; it
    supplies no device layout, kernel, numerical policy or expected value.
    """

    def __init__(self, *, facts_path: Path, policy) -> None:
        from merlin.targetgen.contract.readback_policy import COHERENT_DUMP_V1, ReadbackPolicy

        if type(policy) is not ReadbackPolicy or policy.transport != COHERENT_DUMP_V1:
            raise ValueError("neutral memory reader requires explicit coherent full values")
        self.facts_path, self.policy, self._prepared = Path(facts_path), policy, None

    def _admit(self, cb, backend, simulator, elf_path):
        import math

        from merlin.runtime.backends import base as backends
        from merlin.runtime.backends.chipyard_rocc_bound import BoundRoCCBackend
        from merlin.runtime.harness_render import _roster, _tensor

        from .native_output_readback import _elf_output_symbols

        if (
            type(backend) is not BoundRoCCBackend or simulator != "gsim"
            or not _same_selected_backend(backend, backends.get_backend(backend.target), simulator)
            or backend.engine_readback_policy(simulator) != self.policy
            or backend.memory_readback_transport(simulator) != "gsim_coherent_dump_v1"
        ):
            raise ValueError("neutral memory reader differs from the original selected engine")
        if (
            self.facts_path.resolve() != self.facts_path
            or any(p.is_symlink() for p in (self.facts_path, *self.facts_path.parents))
            or not self.facts_path.is_file()
        ):
            raise ValueError("neutral memory reader has no original ordinary facts")
        facts_raw = self.facts_path.read_bytes()
        if canonical_json(json.loads(facts_raw)) != canonical_json(backend._data()["facts"]):
            raise ValueError("neutral memory reader facts differ from selected execution")
        normalized, slots = _roster(cb)
        layouts = {name: _tensor(spec) for name, spec in normalized["tensors"].items()}
        outputs = []
        for index, name in enumerate(slots):
            if name not in normalized["kernel_abi"]["outputs"]:
                continue
            shape, dtype, extent = layouts[name]
            physical_dtype = normalized["tensors"][name]["dtype"]
            if not shape or physical_dtype not in {"i8", "i16", "i32", "i64", "f32"}:
                raise ValueError("neutral memory reader needs a supported dense non-scalar output")
            strides = [math.prod(shape[axis + 1:]) for axis in range(len(shape))]
            outputs.append({
                "tensor": name, "dtype": physical_dtype, "logical_shape": list(shape),
                "logical_strides_elements": strides, "offset_elements": 0,
                "physical_word_bytes": dtype.element_bits // 8,
                "physical_word_kind": "float_code" if dtype.is_float else "signed_integer",
                "symbol": f"tensor_{index}", "bytes": extent,
            })
        if not outputs or len(outputs) > 1024 or sum(row["bytes"] for row in outputs) > 256 * 1024 * 1024:
            raise ValueError("neutral memory reader exceeds its complete bounded output roster")
        elf_path = Path(elf_path)
        if elf_path.resolve() != elf_path or any(p.is_symlink() for p in (elf_path, *elf_path.parents)):
            raise ValueError("neutral memory reader requires a canonical original ELF")
        raw = elf_path.read_bytes()
        if len(raw) < 18 or raw[16:18] != b"\x02\x00":
            raise ValueError("neutral memory reader requires a fixed-address ET_EXEC ELF")
        symbols, _ = _elf_output_symbols(raw, {row["symbol"]: row["bytes"] for row in outputs})
        for row in outputs:
            row["address"] = symbols[row["symbol"]]
            if row["address"] % row["physical_word_bytes"]:
                raise ValueError("neutral output object is misaligned")
        if elf_path.read_bytes() != raw or self.facts_path.read_bytes() != facts_raw:
            raise ValueError("neutral output source changed during admission")
        return {
            "schema": "merlin.neutral_coherent_output.v1", "elf_sha256": _sha(raw),
            "command_buffer_sha256": _sha(canonical_json(cb)), "rtl_facts_sha256": _sha(facts_raw),
            "backend_selection_sha256": _sha(backend._selection_json.encode()),
            "engine": backend.execution_engine_identity(simulator), "policy": self.policy.record(),
            "outputs": outputs,
        }

    def prepare(self, *, cb, target, elf_path, workdir, simulator, backend):
        if self._prepared is not None or type(cb) is not dict or target != backend.target:
            raise ValueError("neutral memory reader needs one original selected invocation")
        admission = self._admit(cb, backend, simulator, elf_path)
        workdir = Path(workdir)
        if (
            workdir.resolve() != workdir or any(p.is_symlink() for p in (workdir, *workdir.parents))
            or not workdir.is_dir()
        ):
            raise ValueError("neutral memory reader requires a canonical run-owned workdir")
        source = Path(tempfile.mkdtemp(prefix="neutral_readback_", dir=workdir))
        cb_path, regions_path, output_path = (
            source / "command_buffer.json", source / "regions.txt", source / "output.dump"
        )
        cb_raw = canonical_json(cb)
        regions = [{"base": row["address"], "bytes": row["bytes"]}
                   for row in sorted(admission["outputs"], key=lambda row: row["address"])]
        regions_raw = "".join(f"{row['base']:#x} {row['bytes']}\n" for row in regions).encode("ascii")
        for path, raw in ((cb_path, cb_raw), (regions_path, regions_raw)):
            with path.open("xb") as output:
                output.write(raw)
        if (
            _sha(cb_raw) != admission["command_buffer_sha256"] or canonical_json(cb) != cb_raw
            or cb_path.read_bytes() != cb_raw or regions_path.read_bytes() != regions_raw
        ):
            raise ValueError("neutral memory request changed during original preparation")
        request = {
            "schema": "oracle_memory_readback_v1", "transport": "gsim_coherent_dump_v1",
            "elf_sha256": admission["elf_sha256"], "regions": regions,
            "regions_path": str(regions_path), "output_path": str(output_path),
        }
        self._prepared = {
            "cb": cb, "cb_raw": cb_raw, "cb_path": cb_path, "regions_path": regions_path,
            "regions_raw": regions_raw, "output_path": output_path, "source": source,
            "target": target, "backend": backend, "simulator": simulator, "elf_path": Path(elf_path),
            "admission": admission, "request": request, "request_raw": canonical_json(request),
            "facts_path": self.facts_path, "policy_record": self.policy.record(),
        }
        return {"memory_readback": request}

    def decode(self, console):
        from merlin.perf.whole_model_gsim import read_dump
        from merlin.targetgen.contract.readback_policy import require_memory_completion, require_memory_value_roster

        from .native_output_readback import _decode_physical_regions

        state = self._prepared
        if state is None or type(console) is not str:
            raise ValueError("neutral memory readback has no original prepared invocation")
        require_memory_completion(console, {})
        if self.facts_path != state["facts_path"] or self.policy.record() != state["policy_record"]:
            raise ValueError("neutral memory readback facts or policy selection changed")
        current = self._admit(state["cb"], state["backend"], state["simulator"], state["elf_path"])
        if (
            canonical_json(current) != canonical_json(state["admission"])
            or canonical_json(state["cb"]) != state["cb_raw"]
            or canonical_json(state["request"]) != state["request_raw"]
            or state["cb_path"].is_symlink() or state["cb_path"].read_bytes() != state["cb_raw"]
            or state["regions_path"].is_symlink() or state["regions_path"].read_bytes() != state["regions_raw"]
            or state["source"].is_symlink() or not state["source"].is_dir()
        ):
            raise ValueError("neutral memory readback source or request changed")
        rows = current["outputs"]
        expected = sorted((row["address"], row["bytes"]) for row in rows)
        path = state["output_path"]
        if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError("neutral memory output is not an ordinary complete dump")
        expected_bytes = 48 + 16 * len(rows) + sum(row["bytes"] for row in rows)
        raw = path.read_bytes()
        if len(raw) != expected_bytes:
            raise ValueError("neutral memory dump omits or adds output bytes")
        parsed = read_dump(path, expected)
        if parsed["source"] != "coherent" or path.read_bytes() != raw:
            raise ValueError("neutral memory dump is not stable coherent output")
        outputs = _decode_physical_regions(rows, parsed["regions"])
        require_memory_value_roster(state["cb"], outputs)
        after = self._admit(state["cb"], state["backend"], state["simulator"], state["elf_path"])
        if canonical_json(after) != canonical_json(current) or path.is_symlink() or path.read_bytes() != raw:
            raise ValueError("neutral memory readback changed during decoding")
        return outputs, {
            "schema": "merlin.neutral_coherent_readback.v1", "status": "complete", "source": "coherent",
            "scope": "complete original dense values only; numeric and phase qualification remain caller obligations",
            "transport": "gsim_coherent_dump_v1", "elf_sha256": current["elf_sha256"],
            "layout_admission_sha256": _sha(canonical_json(current)), "artifact_sha256": _sha(raw),
            "request_sha256": _sha(state["request_raw"]), "command_buffer_sha256": _sha(state["cb_raw"]),
            "console_sha256": _sha(console.encode()), "output_path": str(path),
            "outputs": [{"name": row["tensor"], "shape": row["logical_shape"], "bytes": row["bytes"]} for row in rows],
        }


def execute_memory_elf(
    *,
    cb: dict,
    target: str,
    elf_path: Path,
    workdir: Path,
    simulator: str,
    backend: Any,
    timeout: int,
    expected_elf_sha256: str,
    facts_path: Path,
    post_run_revalidate: Callable[[], None],
    readback_policy=None,
) -> tuple[str, dict[str, list], dict, dict]:
    """Execute one already-linked ELF through the normal selected backend.

    This helper does not compile, select an engine, grade, or certify. The caller
    owns those checks and must pass the digest from its verified build receipt.
    The backend retains ordinary return-code, assertion and native-slot handling.
    """
    from merlin.common.digest import sha256_file
    from merlin.targetgen.contract.readback_policy import require_memory_completion, require_memory_value_roster

    if (
        type(expected_elf_sha256) is not str
        or len(expected_elf_sha256) != 64
        or any(letter not in "0123456789abcdef" for letter in expected_elf_sha256)
        or not callable(post_run_revalidate)
    ):
        raise ValueError("memory execution has no exact completed-build ELF digest and post-run verifier")
    workdir = Path(workdir)
    if not workdir.is_absolute() or workdir.exists() or workdir.is_symlink():
        raise ValueError("memory execution requires a fresh absolute workdir")
    parent = workdir.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("memory execution parent is absent or indirect")
    workdir.mkdir(mode=0o700, exist_ok=False)
    hook = NativeMemoryReadback(facts_path=facts_path, policy=readback_policy)
    selected = hook.prepare(
        cb=cb,
        target=target,
        elf_path=Path(elf_path),
        workdir=workdir,
        simulator=simulator,
        backend=backend,
    )
    if (
        type(selected) is not dict
        or set(selected) != {"memory_readback"}
        or type(selected["memory_readback"]) is not dict
        or selected["memory_readback"].get("elf_sha256") != expected_elf_sha256
    ):
        raise ValueError("memory preflight differs from the completed-build ELF")
    if sha256_file(elf_path) != expected_elf_sha256:
        raise ValueError("memory execution ELF changed after preflight")
    try:
        console = backend.run_elf(
            elf_path, simulator=simulator, timeout=timeout, memory_readback=selected["memory_readback"]
        )
    except Exception as exc:  # noqa: BLE001 -- preserve available raw subprocess evidence, then re-raise
        _archive_if_present(workdir / "console.partial", getattr(exc, "stdout", None))
        _archive_if_present(workdir / "stderr.partial", getattr(exc, "stderr", None))
        raise
    _archive_if_present(workdir / "console.txt", console)
    if sha256_file(elf_path) != expected_elf_sha256:
        raise ValueError("memory execution ELF changed during native run")
    post_run_revalidate()
    if type(console) is not str:
        raise ValueError("memory execution requires a complete text console")
    serial_outputs, metrics = backend.parse_output(console)
    require_memory_completion(console, serial_outputs)
    outputs, evidence = hook.decode(console)
    require_memory_value_roster(cb, outputs)
    if sha256_file(elf_path) != expected_elf_sha256:
        raise ValueError("memory execution ELF changed after native run")
    if type(metrics) is not dict or type(evidence) is not dict or evidence.get("status") != "complete":
        raise ValueError("memory execution returned incomplete metrics or output admission")
    return console, outputs, metrics, {**evidence, "console_path": str(workdir / "console.txt")}
