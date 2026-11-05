"""Private integration controls for actual harmless joined subprocess records."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

from merlin.common import invocation_record as I
from merlin.runtime.backends import rtl_build_receipt as V


def pin(p):
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def check(p):
    q = Path(p["path"])
    if not q.is_file() or any(x.is_symlink() for x in (q, *q.parents)) or pin(q) != p:
        raise ValueError("selected pin changed")
    return q


@pytest.fixture
def chain(tmp_path):
    def member(name):
        p = tmp_path / name
        p.write_text(name)
        return p

    source = member("source.fir")
    transport = member("public_transport.cc")
    bridge = member("public_bridge.cc")
    driver = member("driver.cc")
    dependency = member("tool_dependency")
    stages = []

    def step(role, inputs, names):
        outputs = [tmp_path / n for n in names]
        code = "from pathlib import Path;import sys\nfor n in sys.argv[1:]:Path(n).write_text(n)\n"
        directory = tmp_path / role
        I.run(
            [sys.executable, "-I", "-B", "-c", code, *map(str, outputs)],
            directory=directory,
            stage=role,
            cwd=tmp_path,
            env={"LC_ALL": "C"},
            inputs=inputs,
            outputs=outputs,
            dependencies=(dependency,),
            capture_output=True,
            timeout=10,
        )
        invocation = next(directory.rglob("invocation.json"))
        I.require_environment(invocation, environment={"LC_ALL": "C"})
        stages.append({"role": role, "invocation": pin(invocation), "environment": {"LC_ALL": "C"}})
        return outputs

    hdl = step(V.ROLES[0], [source], ["selected.sv"])
    model = step(V.ROLES[1], hdl, ["model.cc"])
    archives = step(V.ROLES[2], model, ["model.a"])
    config = step(V.ROLES[3], [transport], ["config.h"])
    library = step(V.ROLES[4], [transport, *config], ["transport.a"])[0]
    binary = step(V.ROLES[5], [*archives, library, bridge, driver], ["executable"])[0]
    doc = {
        "schema": V.SCHEMA,
        "engine": "verilator",
        "target": "synthetic",
        "config": "SyntheticConfig",
        "firrtl": pin(source),
        "binary": pin(binary),
        "hdl": list(map(pin, hdl)),
        "model_sources": list(map(pin, model)),
        "model_archives": list(map(pin, archives)),
        "transport_sources": [pin(transport)],
        "transport_configuration": list(map(pin, config)),
        "transport_library": pin(library),
        "bridge_sources": [pin(bridge)],
        "driver_source": pin(driver),
        "stages": stages,
        "memory_policy": {"response": "fixed"},
        "reset_policy": {"pulse": 8},
    }
    receipt = tmp_path / "receipt.json"

    def validate(value=doc, **kw):
        receipt.write_text(json.dumps(value))
        return V.validate(
            target="synthetic",
            config="SyntheticConfig",
            original_fir_sha256=pin(source)["sha256"],
            binary=kw.get("binary", binary),
            receipt=receipt,
            checked_pin=check,
        )

    return doc, validate, dependency, tmp_path


def test_observed_chain_binds_only_source_and_build_identity(chain):
    doc, validate, _, _ = chain
    identity = validate()
    assert identity["binary"] == doc["binary"]
    assert "execution" in identity["scope"] and "unqualified" in identity["scope"]


@pytest.mark.parametrize(
    "fault",
    [
        "target",
        "config",
        "fir",
        "binary_hash",
        "unrelated_stage",
        "stage_order",
        "environment",
        "join",
        "missing_policy",
        "unknown_field",
    ],
)
def test_mismatched_or_incomplete_receipt_refuses(chain, fault):
    doc, validate, _, tmp = chain
    doc = copy.deepcopy(doc)
    if fault in ("target", "config"):
        doc[fault] = "another"
    elif fault == "fir":
        other = tmp / "other.fir"
        other.write_text("other")
        doc["firrtl"] = pin(other)
    elif fault == "binary_hash":
        doc["binary"]["sha256"] = "0" * 64
    elif fault == "unrelated_stage":
        doc["stages"][0]["invocation"] = doc["stages"][1]["invocation"]
    elif fault == "stage_order":
        doc["stages"][0], doc["stages"][1] = doc["stages"][1], doc["stages"][0]
    elif fault == "environment":
        doc["stages"][0]["environment"] = {"LC_ALL": "changed"}
    elif fault == "join":
        other = tmp / "other.sv"
        other.write_text("other")
        doc["hdl"] = [pin(other)]
    elif fault == "missing_policy":
        doc["reset_policy"] = {}
    elif fault == "unknown_field":
        doc["unexpected"] = "value"
    with pytest.raises(ValueError):
        validate(doc)


def test_equal_byte_sibling_cannot_replace_selected_executable(chain):
    doc, validate, _, tmp = chain
    other = tmp / "sibling"
    other.write_bytes(Path(doc["binary"]["path"]).read_bytes())
    with pytest.raises(ValueError, match="another executable"):
        validate(binary=other)


def test_observed_dependency_drift_refuses(chain):
    _, validate, dependency, _ = chain
    dependency.write_text("changed")
    with pytest.raises(ValueError):
        validate()


def test_changed_produced_bridge_refuses_even_when_receipt_is_rehashed(chain):
    doc, validate, _, _ = chain
    bridge = Path(doc["bridge_sources"][0]["path"])
    bridge.write_text("changed")
    doc["bridge_sources"][0] = pin(bridge)
    with pytest.raises(ValueError):
        validate()
