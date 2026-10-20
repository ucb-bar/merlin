"""The selected coherent-memory output decoder, run as one observed invocation.

:func:`merlin.targetgen.contract.compile.run_on_oracle` calls this when an explicit execution
service owns the run: the decoder's return and any payload it declares are published as private,
digest-bound products beside the oracle console. Nothing here names a target.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def observed_memory_decode(reader, console, *, cb, elf, workdir, policy, dependencies=()):
    """Retain the actual selected decoder and its declared private products.

    A callback return or payload pin supplies attribution only. Original packet
    membership, observer integrity and execution semantics remain independent.
    """
    from merlin.common import invocation_record

    from .readback_policy import BUILD_RECEIPT, require_memory_value_roster

    work = Path(workdir).resolve()
    product = work / "readback_decode.json"
    if product.exists() or product.is_symlink():
        raise ValueError("memory decoder product already exists in its private execution owner")
    with invocation_record.observe_call(
        work,
        stage="coherent_memory_decode",
        function=reader.decode,
        arguments={"readback_policy": policy.record()},
        inputs=(Path(elf), work / "oracle_console.log", work / BUILD_RECEIPT),
        outputs=(product,),
        dependencies=(Path(__file__), Path(__file__).with_name("compile.py"), *dependencies),
    ) as observed:
        outputs, evidence = reader.decode(console)
        if type(outputs) is not dict or type(evidence) is not dict or evidence.get("status") != "complete":
            raise ValueError("memory output reader returned no completed full-value admission")
        require_memory_value_roster(cb, outputs)
        declared_payload = evidence.get("payload")
        if declared_payload is not None:
            if type(declared_payload) is not dict or set(declared_payload) != {"path", "sha256"}:
                raise ValueError("memory decoder declared an unsupported payload product")
            payload = Path(declared_payload["path"])
            if (
                not payload.is_absolute()
                or payload.resolve() != payload
                or any(path.is_symlink() for path in (payload, *payload.parents))
                or not payload.is_relative_to(work)
                or not payload.is_file()
            ):
                raise ValueError("memory decoder payload escapes its private execution owner")
            if hashlib.sha256(payload.read_bytes()).hexdigest() != declared_payload["sha256"]:
                raise ValueError("memory decoder payload differs from its actual declared bytes")
            observed.outputs = (*observed.outputs, payload)
        encoded = (
            json.dumps(
                {
                    "outputs": outputs,
                    "memory_evidence": evidence,
                    "scope": "actual decoder return and declared products only; observer/runtime/effects UNKNOWN",
                },
                sort_keys=True,
            )
            + "\n"
        )
        # The selected observer runs before publication. Exclusive creation also
        # refuses a file or symlink it creates after the earlier metadata check.
        with product.open("x", encoding="utf-8") as output:
            output.write(encoded)
        observed.returned(stdout=encoded)
    invocation_record.verify(observed.path)
    return (
        outputs,
        evidence,
        {
            "record": {"path": str(observed.path), "sha256": hashlib.sha256(observed.path.read_bytes()).hexdigest()},
            "product": {"path": str(product), "sha256": hashlib.sha256(product.read_bytes()).hexdigest()},
            "scope": "source-bound decoder invocation only; no observer, stage, hardware or timer authority",
        },
    )
