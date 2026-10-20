"""Verify observed RTL build identity; execution and correctness remain separate."""

from __future__ import annotations

import json
from pathlib import Path

from merlin.common import invocation_record as I

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
