"""Opt-in structural interfaces for shared passes, without qualification authority.

A host pins this requirement and its library separately. Package declarations do
not grant effects, numerical legality, import access or executable consumption.
The ordinary runner must still verify the selected IR reaches its real consumer.
"""

from __future__ import annotations

import json

FAMILY_NEEDS = {
    "producer_packing": ("scalar_semantics", "layout_maps", "ownership"),
    "horizontal_matmul_fusion": ("ordered_reductions", "operand_reuse", "persistent_state", "completion"),
    "multi_output_fusion": ("scalar_semantics", "shared_generation", "ownership", "completion"),
    "bounded_recomputation": ("scalar_semantics", "repeatability", "fault_observations", "stable_inputs"),
    "joint_layout_fusion": ("layout_maps", "scalar_semantics", "ownership", "fault_observations"),
    "padding_gather": ("scalar_semantics", "layout_maps", "bounds", "fault_observations"),
    "destination_forwarding": ("ownership", "alias_lifetime", "completion"),
    "dma_coalescing": ("physical_ranges", "alias_lifetime", "completion", "fault_observations"),
    "dead_local_writes": ("physical_ranges", "read_write_effects", "completion", "fault_observations"),
}
V2_FAMILY_NEEDS = {
    **FAMILY_NEEDS,
    "buffering_pipelining": (
        "physical_ranges",
        "alias_lifetime",
        "dependency_scheduling",
        "temporary_allocation",
        "completion",
        "resource_bounds",
    ),
}


def _families(version):
    if type(version) is not int or version not in (1, 2):
        raise ValueError("shared pass contract version must be explicitly 1 or 2")
    return FAMILY_NEEDS if version == 1 else V2_FAMILY_NEEDS


FOUNDATIONS = (
    "ordered_reductions",
    "operand_reuse",
    "temporary_allocation",
    "dependency_scheduling",
    "buffering_pipelining",
    "persistent_configuration",
    "final_program_limits",
)
_CONTEXT = ("hardware", "numerical_policy", "resources", "execution", "library")
_BOUNDARIES = ("computation", "device", "emission")
_FOUNDATION_BOUNDARIES = {
    "computation": ("ordered_reductions", "operand_reuse"),
    "device": ("temporary_allocation", "dependency_scheduling", "buffering_pipelining", "persistent_configuration"),
    "emission": ("final_program_limits",),
}
_PIPELINE = {
    "source_sha256": True,
    "context": list(_CONTEXT),
    "requires_and_invalidates": True,
    "bounded_repair": True,
    "unknown": "retain_original",
}


def shared_pass_requirement(*, required_supported=(), version=1) -> dict:
    """Return detached structural requirements; this does not select a library.

    ``required_supported`` is original host policy. Unknown/unsupported family
    declarations cannot satisfy that policy, and none can establish qualification.
    The ceilings bound declarations rather than price or authorize an optimization.
    """
    families = _families(version)
    if type(required_supported) not in (tuple, list):
        raise ValueError("required_supported must be an ordered family sequence")
    if any(type(name) is not str or name not in families for name in required_supported) or len(
        set(required_supported)
    ) != len(required_supported):
        raise ValueError("required_supported contains duplicate or unknown families")
    return {
        "schema": f"merlin.shared_pass_requirement.v{version}",
        "families": {name: list(needs) for name, needs in families.items()},
        "foundations": list(FOUNDATIONS),
        "boundaries": list(_BOUNDARIES),
        "pipeline": json.loads(json.dumps(_PIPELINE)),
        "limits": {"max_growth_ops": 65536, "max_repairs": 2},
        "required_supported": list(required_supported),
    }


def _object(value, fields, label):
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError(f"{label} requires exactly {', '.join(fields)}")
    return value


def _text(value, label):
    if type(value) is not str or not value.strip() or len(value) > 1024:
        raise ValueError(f"{label} must be a bounded nonempty string")


def _names(value, allowed, label, *, complete=False):
    if (
        type(value) is not list
        or len(value) > len(allowed)
        or any(type(name) is not str or name not in allowed for name in value)
        or len(set(value)) != len(value)
        or (complete and set(value) != set(allowed))
    ):
        raise ValueError(f"{label} requires unique known names" + (" and complete coverage" if complete else ""))


def _integer(value, ceiling, label):
    if type(value) is not int or not 0 <= value <= ceiling:
        raise ValueError(f"{label} must be an integer in [0, {ceiling}]")


def _requirement(value):
    fields = ("schema", "families", "foundations", "boundaries", "pipeline", "limits", "required_supported")
    _object(value, fields, "shared pass requirement")
    schema = value["schema"]
    if type(schema) is not str:
        raise ValueError("unknown shared pass requirement schema")
    version = {"merlin.shared_pass_requirement.v1": 1, "merlin.shared_pass_requirement.v2": 2}.get(schema)
    if version is None:
        raise ValueError("unknown shared pass requirement schema")
    expected = shared_pass_requirement(required_supported=value["required_supported"], version=version)
    for field in fields:
        if field != "limits" and json.dumps(value[field], sort_keys=True) != json.dumps(
            expected[field], sort_keys=True
        ):
            raise ValueError(f"unknown or malformed shared pass requirement {field}")
    _object(value["limits"], expected["limits"], "requirement limits")
    for name, ceiling in expected["limits"].items():
        _integer(value["limits"][name], ceiling, name)


def validate_shared_pass_requirement(value) -> dict:
    """Validate and detach an original structural requirement, issuing no grants."""
    _requirement(value)
    return json.loads(json.dumps(value))


def _command(manifest, name):
    _text(name, "boundary command")
    commands = manifest.get("commands")
    command = commands.get(name) if type(commands) is dict else None
    _object(command, ("argv",), f"command {name}")
    argv = command["argv"]
    if type(argv) is not list or not 1 <= len(argv) <= 128:
        raise ValueError("boundary command argv must be bounded and nonempty")
    for arg in argv:
        _text(arg, "command argument")
    if not any("{input_mlir}" in arg for arg in argv):
        raise ValueError(f"command {name} must consume selected IR through {{input_mlir}}")
    return command


def validate_target_ir_consumer_declaration(manifest) -> dict:
    """Inspect public ABI transport declarations, without package/runtime access.

    This does not resolve or execute argv, verify changed-IR consumption, or grant
    instruction, effect or numerical authority. Protected runners reuse this
    pure structure check and retain their own enforcement and error categories.
    """
    expected = {
        "schema": "merlin.target_ir_consumer.v1",
        "command": "emit_target_bundle",
        "input": "selected_target_mlir",
        "output": "command_buffer_and_target_artifact",
    }
    manifest = manifest if type(manifest) is dict else {}
    declaration = manifest.get("target_ir_consumer")
    commands = manifest.get("commands")
    command = commands.get("emit_target_bundle") if type(commands) is dict else None
    argv = command.get("argv") if type(command) is dict else None
    if (
        manifest.get("abi_version", "0.1") != "0.1"
        or type(declaration) is not dict
        or declaration != expected
        or type(command) is not dict
        or set(command) != {"argv"}
        or type(argv) is not list
        or not argv
        or any(type(value) is not str or not value for value in argv)
        or not all(any(placeholder in value for value in argv) for placeholder in ("{input_mlir}", "{output_json}"))
    ):
        raise ValueError(
            "selected target configuration requires ABI 0.1 and an explicit "
            "merlin.target_ir_consumer.v1 emit_target_bundle consuming "
            "{input_mlir} and producing {output_json} plus the target artifact"
        )
    return json.loads(json.dumps({"declaration": declaration, "command": command, "target": manifest.get("target")}))


def validate_shared_pass_contract(manifest, *, requirement=None) -> dict | None:
    """Check declarations only, returning a detached record.

    No requirement preserves legacy behavior without inspecting the package.
    Explicit opt-in never downgrades a missing/malformed declaration. ``supported``
    means the package promises an interface; original laws and consumed-program
    checks remain required. Unknown integration is not hardware non-support.
    """
    if requirement is None:
        return None
    _requirement(requirement)
    if type(manifest) is not dict:
        raise ValueError("selected shared passes require a package manifest")
    _text(manifest.get("target"), "package target")
    value = manifest.get("shared_pass_contract")
    _object(value, ("schema", "boundaries", "pipeline", "controls", "families"), "shared_pass_contract")
    version = 1 if requirement["schema"] == "merlin.shared_pass_requirement.v1" else 2
    families = _families(version)
    if value["schema"] != f"merlin.shared_pass_contract.v{version}":
        raise ValueError("unknown shared pass contract schema")
    boundaries = _object(value["boundaries"], _BOUNDARIES, "IR boundaries")
    vocabulary = set(FOUNDATIONS).union(*(set(needs) for needs in families.values()))
    commands = {}
    for name, boundary in boundaries.items():
        _object(boundary, ("ir", "command", "contracts", "features"), f"{name} boundary")
        _text(boundary["ir"], "named IR")
        commands[name] = _command(manifest, boundary["command"])
        contract_fields = ("semantics", "effects", "capabilities")
        if version == 2:
            contract_fields += ("resources", "completion")
        contracts = _object(boundary["contracts"], contract_fields, "boundary contracts")
        for identity in contracts.values():
            _text(identity, "contract identity")
        _names(boundary["features"], vocabulary, "boundary features")
    if len({boundary["ir"] for boundary in boundaries.values()}) != len(_BOUNDARIES):
        raise ValueError("IR boundaries must have distinct explicit names")
    for name, features in _FOUNDATION_BOUNDARIES.items():
        if not set(features) <= set(boundaries[name]["features"]):
            raise ValueError(f"{name} boundary features omit foundational interfaces")
    device = boundaries["device"]
    if device["ir"] != "selected_target_mlir" or device["command"] != "emit_target_bundle":
        raise ValueError("device boundary must reuse merlin.target_ir_consumer.v1")
    target_consumer = validate_target_ir_consumer_declaration(manifest)
    if type(value["pipeline"]) is not dict or json.dumps(value["pipeline"], sort_keys=True) != json.dumps(
        _PIPELINE, sort_keys=True
    ):
        raise ValueError("pipeline requires current source/context, prerequisites and transitive invalidation")
    rows = value["families"]
    if type(rows) is not list or len(rows) != len(families):
        raise ValueError("families require one status row per mechanism")
    by_name = {}
    for row in rows:
        _object(row, ("name", "status", "reason", "boundary", "requirements"), "family status")
        name = row["name"]
        if type(name) is not str or name not in families or name in by_name:
            raise ValueError("duplicate or unknown family")
        if row["status"] not in ("supported", "unsupported", "unknown"):
            raise ValueError("family status must be supported, unsupported or unknown")
        _text(row["reason"], "family status reason")
        if type(row["boundary"]) is not str or row["boundary"] not in boundaries:
            raise ValueError("family references an unknown boundary")
        _names(row["requirements"], families[name], "family requirements", complete=True)
        if row["status"] == "supported" and not set(row["requirements"]) <= set(
            boundaries[row["boundary"]]["features"]
        ):
            raise ValueError("supported family lacks declared boundary requirements")
        by_name[name] = row
    for name in requirement["required_supported"]:
        if by_name[name]["status"] != "supported":
            raise ValueError(f"original requirement needs supported interface: {name}")
    controls = _object(value["controls"], ("default_policy", "choices", "max_repairs", "retain_original"), "controls")
    if controls["retain_original"] is not True:
        raise ValueError("controls must retain the original alternative")
    _integer(controls["max_repairs"], requirement["limits"]["max_repairs"], "max_repairs")
    _names(controls["default_policy"], families, "default_policy")
    supported = {name for name, row in by_name.items() if row["status"] == "supported"}
    if set(controls["default_policy"]) != supported:
        raise ValueError("default_policy must cover every supported family exactly once")
    choices = _object(controls["choices"], families, "bounded choices")
    previous = -1
    for name in controls["default_policy"]:
        row = by_name[name]
        stage = _BOUNDARIES.index(row["boundary"])
        if row["status"] != "supported" or stage < previous:
            raise ValueError("default_policy requires supported interfaces in boundary order")
        previous = stage
    for name, choice in choices.items():
        _object(choice, ("enabled", "max_growth_ops"), "family choice")
        # Exact Boolean lists, not integers comparing equal to False/True.
        enabled = choice["enabled"]
        if type(enabled) is not list or any(type(flag) is not bool for flag in enabled):
            raise ValueError("enabled choices must be explicit Booleans")
        if enabled != [False] and (enabled != [False, True] or by_name[name]["status"] != "supported"):
            raise ValueError("only supported interfaces may offer enabled choices; original must remain")
        if name in controls["default_policy"] and enabled != [False, True]:
            raise ValueError("default_policy selects a disabled mechanism")
        _integer(choice["max_growth_ops"], requirement["limits"]["max_growth_ops"], "max_growth_ops")
    return json.loads(json.dumps({"contract": value, "commands": commands, "target_consumer": target_consumer}))
