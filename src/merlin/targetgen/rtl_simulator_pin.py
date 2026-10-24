"""Opt-in byte pinning for an operator-selected Chipyard RTL simulator."""

from __future__ import annotations

import os
from pathlib import Path

from merlin.common import provenance


def _commits(checked) -> dict[str, str]:
    commits = {checked.pin: checked.observed.commit}
    for covered in checked.covered:
        commits.update(_commits(covered))
    return commits


def _verify_facts(pin, target: str, checkout: Path) -> tuple[Path, Path]:
    from .rtl.facts import find_facts, load_facts
    from .rtl.source_selection import load_selection, production_consistency
    from .target_experiment import load_capability_manifest

    facts = load_facts(target)
    inputs = facts.get("inputs") or {}
    selected = load_selection(inputs["source_bundle_path"], target=target)
    if selected["selection_sha256"] != inputs.get("source_bundle_sha256"):
        raise ValueError("RTL facts source bundle changed")
    if production_consistency(selected)["status"] != "verified":
        raise ValueError("RTL facts lack verified source production")
    config = (load_capability_manifest(target).contract.get("runtime") or {}).get("rtl_sim_config")
    if not config or config != selected["config"]:
        raise ValueError("RTL facts configuration differs from the selected runtime")
    firrtl = selected["sources"]["firrtl"]
    if inputs.get("fir_sha256") != firrtl["sha256"]:
        raise ValueError("RTL facts input differs from the selected FIRRTL")
    relative = Path(firrtl["path"]).resolve().relative_to(checkout.resolve()).as_posix()
    if dict(pin.build_products).get(relative) != firrtl["sha256"]:
        raise ValueError("RTL facts FIRRTL is not a byte-checked product of the simulator pin")
    return Path(find_facts(target)), Path(inputs["source_bundle_path"])


def _selected_product(backend, engine: str, pin) -> tuple[Path, Path]:
    resolver = getattr(backend, f"{engine}_path", None)
    if not callable(resolver):
        raise ValueError(f"backend cannot resolve its {engine!r} executable for byte verification")
    executable = Path(resolver()).resolve(strict=True)
    checkout = pin.checkout()
    if checkout is None:
        raise ValueError(f"RTL simulator pin {pin.name!r} has no checkout")
    relative = executable.relative_to(checkout.resolve()).as_posix()
    if relative not in dict(pin.build_products):
        raise ValueError(f"selected executable is not a build product of RTL simulator pin {pin.name!r}")
    return executable, checkout


def verify_selected(backend, engine: str, target: str) -> dict | None:
    """Require the actual backend executable to be an unconditionally checked product.

    A pin constrains an explicit engine choice. It cannot select a simulator by
    directory name or silently authorize a different backend's executable.
    """
    name = os.environ.get("MERLIN_RTL_SIM_PIN", "").strip()
    if not name:
        return None
    if os.environ.get("MERLIN_REQUIRED_RTL_ENGINE", "").strip() != engine:
        raise ValueError("MERLIN_RTL_SIM_PIN requires the matching MERLIN_REQUIRED_RTL_ENGINE")
    pin = provenance.load_pins()[name]
    if target not in pin.targets:
        raise ValueError(f"RTL simulator pin {name!r} does not declare target {target!r}")
    executable, checkout = _selected_product(backend, engine, pin)
    # Do not reuse a grade's observation cache: build products can be replaced
    # between selection and execution even when the source commit is unchanged.
    checked = provenance.require(name)
    sources = _verify_facts(pin, target, checkout)
    record = provenance.record(pins={name: checked}, sources=sources, artifacts={"simulator": executable})
    record["execution_identity"] = {
        "pin": name,
        "commits": _commits(checked),
        "build_products": dict(pin.build_products),
        "simulator_path": str(executable),
        "simulator_sha256": record["artifact_digests"]["simulator"],
        "facts_source_digest": record["source_digest"],
    }
    return record
