"""Inert trusted-host prototype for one graph service → compiler handoff.

The trusted caller selects a live binding, the real service and a compiler
callback. Only request data crosses consume(). No context is constructed, no
native hold is released, and no serialized report recreates authority. This
entry point cannot constrain compiler calls made elsewhere by the host.
"""

from __future__ import annotations

from pathlib import Path

from merlin.verify.lean_graph_pipeline_gate_service import GraphServiceBinding, validate_binding

from .common_transforms import CommonTransformService


class ConsumerRefused(ValueError):
    """The chosen consumer path has no eligible exact live service delivery."""


class GraphConsumerBridge:
    """Trusted wiring only; compiler must consume exactly the supplied IR text.

    Own the instance/request/service exclusively during consume(). Callbacks
    are trusted host capabilities, never request fields. This prototype does
    not claim atomicity against other Python threads changing host objects.
    """

    __slots__ = ("_service", "_binding", "_compiler")

    def __init__(self, *, service, binding, compiler):
        if type(service) is not CommonTransformService or type(binding) is not GraphServiceBinding:
            raise ConsumerRefused("exact selected service and live graph binding required")
        if service.graph_binding is not binding or not callable(compiler):
            raise ConsumerRefused("trusted service/binding/compiler selection differs")
        validate_binding(binding, compiler_profile_sha256=service.profile["sha256"])
        self._service, self._binding, self._compiler = service, binding, compiler

    def consume(self, request):
        """Apply the actual service, then hand off only the SAME eligible dict.

        Refusal/exception propagates before the compiler callback. Each successful
        invocation calls that callback once with the exact final immutable text.
        It does not prove what a callback compiles or constrain other entrypoints.
        """
        if (
            type(request) is not dict
            or set(request) != {"schema", "source_ir", "options"}
            or type(request["source_ir"]) is not str
            or type(request["options"]) is not dict
        ):
            raise ConsumerRefused("closed transformation request data required")
        service, binding = self._service, self._binding
        original = request["source_ir"]
        if service.graph_binding is not binding:
            raise ConsumerRefused("trusted binding changed before service execution")
        document = CommonTransformService.apply(service, request, work_root=Path(binding.workspace))
        if type(document) is not dict or type(document.get("transformed_ir")) is not str:
            raise ConsumerRefused("actual service output dict and final IR required")
        final = document["transformed_ir"]
        if request["source_ir"] != original or service.graph_binding is not binding:
            raise ConsumerRefused("original source or trusted binding changed")
        if CommonTransformService.compiler_eligible(service, document, source_ir=original) is not True:
            raise ConsumerRefused("same live original/final service delivery is not eligible")
        if (
            request.get("source_ir") != original
            or document.get("transformed_ir") != final
            or service.graph_binding is not binding
        ):
            raise ConsumerRefused("source/final/binding changed during eligibility revalidation")
        return self._compiler(final)
