"""The runtime layer of a generated target repo — the SHARED package shape, not a second one.

This module used to emit its own thing: a ``runtime/adapter/adapter.py`` holding a ``RuntimeAdapter``
CLASS with a bespoke ``lower_target_ir_to_runtime`` / ``encode_command_buffer`` / ``run_simulator``
surface, a ``runtime/simulator/semantics.py`` that re-exported Merlin's reference oracle, and a
``command_encoding.yaml``. Four things were wrong with it, and they compounded:

* **Nothing registered it.** The class was not a ``Backend``, and no ``plugin`` block named it, so the
  whole layer was unreachable through ``runtime.backends.base`` — ``get_backend`` raised ``KeyError``
  for every target this pipeline has ever generated.
* **A different shape from the protocol.** Even reached, a ``RuntimeAdapter`` instance does not satisfy
  the module-level ``Backend`` protocol every other backend in the repo implements.
* **Baked opcodes.** ``RES_PACK`` / ``MATMUL_RESIDENT`` / ``COMMIT`` / ``EVICT`` were written into the
  adapter source, again into ``semantics.py``, and a third time into ``command_encoding.yaml``, beside
  a ``DEFAULT_REQUANT_SHIFT = 4``. Those are facts about one family of machine, emitted into every
  machine's package.
* **A dependency on another target's backend.** ``run_spike`` imported ``runtime.backends.spike`` and
  ran the buffer there, so a generated package's "execution" was a different target's.

So the emission now comes from :mod:`merlin.targetgen.generate.oot_package`, the single place that
says what a generated package looks like — the same code
:func:`merlin.targetgen.capability_manifests.write_oot_target` calls. Two generators materialising two
shapes is not a convergence; it is the same hole with a second entrance.

NOTE FOR THE SANDBOX. This module's path is on
``merlin.targetgen.sandbox.answer_surfaces.ORACLE_CALLABLE_SUBPATHS`` because the old ``semantics.py``
it emitted was a callable route to Merlin's reference oracle. What it emits now contains no such route
— the generated backend declines rather than borrowing the oracle's arithmetic — but the deny entry
stays: removing it is a grant decision, and a grant decision belongs in its own reviewed change.
"""

from __future__ import annotations

from typing import Any

from ...common.artifacts import Artifact, yaml_artifact
from . import oot_package as _pkg


def generate(runtime_adapter_plan: dict[str, Any], target_contract: dict[str, Any] | None = None) -> list[Artifact]:
    """The runtime layer of a generated target repo: the self-registering backend + the work order.

    ``target_contract`` is the synthesized contract for this target; the opcode map is DERIVED from its
    ``encoding`` block and from nothing else. When it is absent or grounds no encoding, the emitted
    ``command_encoding.yaml`` carries ``opcodes: {}`` and an ``UNKNOWN(...)`` derivation naming why, and
    the backend declines — it never substitutes a default.
    """
    target = runtime_adapter_plan.get("target") or (target_contract or {}).get("name") or "target"
    metrics_mapping = runtime_adapter_plan.get("metrics", {"maps_to_common": {}, "target_specific": []})
    return [
        Artifact(f"{_pkg.BACKEND_DIR}/__init__.py", _pkg.backend_module_source(target)),
        yaml_artifact(
            f"{_pkg.BACKEND_DIR}/{_pkg.ENCODING_FILE}",
            _pkg.command_encoding(target, target_contract or {}),
            header=f"GENERATED opcode map for {target} — DERIVED from this target's own contract encoding.",
        ),
        Artifact(_pkg.COMPILER_TOOL, _pkg.compiler_tool_source(target)),
        yaml_artifact(
            _pkg.COMPILER_MANIFEST,
            _pkg.compiler_manifest(target),
            header=(
                f"GENERATED experiment-ABI WORK ORDER for {target} — NOT a compiler. Phase 1 fills in "
                f"the four commands and flips package_capabilities.compiler.provided."
            ),
        ),
        yaml_artifact(
            "runtime/adapter/metrics_mapping.yaml",
            metrics_mapping,
            header="Raw-counter -> common-metric mapping.",
        ),
    ]
