"""Original same-ELF numerical re-audit shared by explicit execution ABIs.

The caller selects storage-name binding; this never changes numerical policy.
"""

import json


def audit_native_numerical_tier(emission, native, *, target, readback_policy, output_values=None):
    from merlin.targetgen.contract.readback_policy import read_console

    from .native_model_execution import (
        NativeModelExecutionError,
        _digest,
        _functional_engine,
        _pinned_native_file,
        _read_pinned_payload,
        checked_source_output_values,
    )

    if output_values is None:
        output_values = checked_source_output_values
    try:
        import yaml

        from merlin.runtime.backends import base as backends
        from merlin.runtime.commandbuffer import declared_output_dtypes
        from merlin.targetgen.capsule_golden import compare
        from merlin.targetgen.golden_store import load_golden

        functional = (native.get("tiers") or {}).get("L2") or {}
        if functional.get("status") not in {"pass", "fail"} or functional.get("engine") != "spike":
            raise NativeModelExecutionError("same candidate ELF has no completed functional L2 execution")
        backend, citation, revalidate = _functional_engine(target)
        if functional.get("engine_citation") != citation:
            raise NativeModelExecutionError("L2 engine citation differs from currently selected bytes")
        revalidate()
        elf = _pinned_native_file(native, "elf")
        if functional.get("elf") != _digest(elf):
            raise NativeModelExecutionError("L2 ran a different ELF")
        console = read_console(_pinned_native_file(functional, "console"), policy=readback_policy)
        observed, _metrics = backend.parse_output(console)
        cb = json.loads(_read_pinned_payload(emission, "command_buffer").decode("utf-8"))
        if readback_policy is not None:
            from merlin.targetgen.contract.readback_policy import require_full_value_roster

            require_full_value_roster(cb, console, observed, policy=readback_policy)
        observed = backends.decode_float_readback(observed, declared_output_dtypes(cb))
        capsule_dir = _pinned_native_file(native["source"], "capsule_declaration").parent
        _pinned_native_file(native["source"], "golden")
        if (capsule_dir / "golden.npz").is_file() != ("golden_arrays" in native["source"]):
            raise NativeModelExecutionError("L2 golden archive presence differs from source pins")
        if "golden_arrays" in native["source"]:
            _pinned_native_file(native["source"], "golden_arrays")
        declaration = yaml.safe_load((capsule_dir / "capsule.yaml").read_text(encoding="utf-8"))
        golden = load_golden(capsule_dir)
        observed = output_values(cb, native, golden, observed)
        if set(observed) != set(golden["outputs"]):
            raise NativeModelExecutionError("L2 console has no complete declared full output")
        numeric = compare(
            golden["outputs"],
            observed,
            declaration["numeric_policy"],
            golden_source=str(golden.get("golden_source") or "independent_capsule"),
        )
        if (
            numeric.get("status") not in {"pass", "fail"}
            or numeric != functional.get("numeric")
            or functional.get("status") != numeric["status"]
        ):
            raise NativeModelExecutionError("L2 numeric receipt differs from independently parsed full output")
        return {
            "status": numeric["status"],
            "engine_citation": citation,
            "elf": _digest(elf),
            "console": _digest(_pinned_native_file(functional, "console")),
            "numeric": numeric,
        }
    except Exception as exc:  # noqa: BLE001 -- unbound engine/console is unavailable, not a tier pass
        return {"status": "unverified", "detail": f"{type(exc).__name__}: {exc}"}
