"""Add ordinary counter wrappers to completed unprofiled program objects.

Only generated host observation code is compiled. All original model, device,
weights and runtime objects remain inputs to the normal recurrent-image link.
"""

import json
from pathlib import Path

from merlin.common import invocation_record
from merlin.llvmlower.group_profile_bindings import emit_profile
from merlin.llvmlower.group_profile_sibling import completed_sibling_bindings

from . import chain_image_layout as layout
from . import chain_image_objects as objects


def prepare(program, directory, *, nm, timeout):
    directory.mkdir()
    model = objects.reopen(program["by_name"]["model.o"])
    undefined = {name for name, kind in objects.symbols(nm, model, timeout=timeout, directory=directory).items()
                 if kind == "U"}
    sidecar = program["receipt"]["output"]["device_sidecar"]
    routing = json.loads(objects.payload(sidecar))
    bindings, observed = completed_sibling_bindings(program, routing, undefined)
    missing = {row["symbol"] for row in routing.get("routed", ())
               if row["symbol"] not in bindings and row["symbol"] not in undefined
               and "_mlir_ciface_" + row["symbol"] not in undefined}
    if missing:
        from .chain_image_profile_descriptor import completed_descriptor_bindings

        descriptor, pins = completed_descriptor_bindings(program, routing, undefined)
        if bindings.keys() & descriptor.keys():
            raise ValueError("completed profile has overlapping original ABI bindings")
        bindings.update(descriptor)
        observed.extend(pins)
    command = layout.command(program, "model.o")
    if len(command["inputs"]) != 1:
        raise ValueError("completed profile requires the actual unique model compiler input")
    original_source = objects.reopen(command["inputs"][0])
    observations = [program["receipt_pin"], program["recipe_pin"], sidecar,
                    program["by_name"]["model.o"], command["inputs"][0], *observed]
    source, flags = emit_profile(directory, routing, routing, undefined, bindings=bindings,
        source=original_source, model_object=model, source_kind="compiled_model_input", observations=observations)
    product = directory / "group_profile.o"
    driver = layout.command(program, "model_main.o")
    invocation_record.run(layout.compile_argv(driver, source, product), directory=directory,
        stage="chain_image_completed_group_profile", inputs=(source,), outputs=(product,),
        dependencies=(Path(__file__), Path(layout.__file__)),
        capture_output=True, text=True, check=True, timeout=timeout)
    selected = objects.pin(product)
    wraps = objects.wrap_names(flags)
    result = {**program, "objects": [*program["objects"], selected],
              "by_name": {**program["by_name"], "group_profile.o": selected},
              "profile_wraps": wraps}
    record = dict(schema="merlin.completed_program_group_profile.v1", source=objects.pin(source), object=selected,
        bindings=objects.pin(directory / "group_profile_bindings.json"), inputs=observations,
        invocations=[objects.pin(p) for p in directory.glob("invocations/*/invocation.json")],
        wraps=list(wraps), original_objects=program["objects"],
        boundary="inclusive external ABI call: bridge, selected leaf and its readout; not pure device time",
        scope="host counter wrapper only; original graph/device/weights sections unchanged")
    return result, record
