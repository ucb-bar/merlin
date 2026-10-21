"""Share the existing session object-symbol isolation across host runtimes."""

import shutil
from pathlib import Path

from merlin.common import invocation_record as I


def namespace_stage_object(source, destination, *, index, entrypoint, nm, objcopy, runner, compilation=None):
    """Rename defined public helpers and their relocations, retaining stage ABI.

    Undefined shared-runtime imports remain unchanged. A device stage must first
    partial-link its host, shim and device definitions so their internal references
    are renamed together. This establishes symbol isolation, not body semantics.
    """
    source, destination = Path(source), Path(destination)
    listing = [nm, "--defined-only", "--extern-only", "--format=posix", source]
    if compilation is None:
        listed = runner(listing)
    else:
        with I.observe_call(
            destination.parent,
            stage="session_symbol_inventory",
            function=runner,
            arguments={"argv": list(map(str, listing))},
            inputs=(source,),
            dependencies=(Path(nm),),
        ) as observed:
            listed = runner(listing)
            observed.returned(stdout=listed.stdout, stderr=listed.stderr)
    preserve = {entrypoint, f"_mlir_ciface_{entrypoint}"}
    symbols = [parts[0] for line in listed.stdout.splitlines() if (parts := line.split()) and parts[0] not in preserve]
    if not symbols:
        if compilation is None:
            shutil.copy2(source, destination)
        else:
            with I.observe_call(
                destination.parent,
                stage="session_symbol_identity",
                function=shutil.copy2,
                arguments={"identity_copy": True},
                inputs=(source,),
            ) as observed:
                shutil.copy2(source, destination)
                observed.outputs = (destination,)
                observed.returned()
        return destination
    command = [
        objcopy,
        *(f"--redefine-sym={symbol}=merlin_stage_{index}__{symbol}" for symbol in symbols),
        source,
        destination,
    ]
    if compilation is None:
        runner(command)
    else:
        with I.observe_call(
            destination.parent,
            stage="session_symbol_namespace",
            function=runner,
            arguments={"argv": list(map(str, command))},
            inputs=(source,),
            dependencies=(Path(objcopy),),
        ) as observed:
            result = runner(command)
            observed.outputs = (destination,)
            observed.returned(stdout=result.stdout, stderr=result.stderr)
    if not destination.is_file():
        raise ValueError(f"stage symbol namespacing produced no object at {destination}")
    return destination
