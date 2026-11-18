"""Prepare the Merlin-owned Interface program before invoking an OOT compiler.

The original request and selected-region ticket remain bound to upstream source.
The backend sees a separately bound Interface request. This adapter records both
identities; parsing, structural checks and receipts are not equivalence evidence.
Providers without an Interface contract retain their explicit existing route.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from merlin.compiler.compilation import Artifact, CompileRequest, Source, request_identity

INTERFACE_IR = "merlin.interface.v1"


def input_contract(provider, request):
    """Select an input contract without asking the provider to parse source."""
    hook = getattr(provider, "input_contract", None)
    contract = hook(request) if callable(hook) else getattr(provider, "input_ir", None)
    if contract not in (None, INTERFACE_IR):
        raise ValueError(f"unsupported backend input contract: {contract!r}")
    return contract


def interface_source(source):
    """Use Merlin's actual conversion; never relabel upstream MLIR as Interface."""
    from merlin.compiler.ir.interface_contract import interface_text, parse_interface
    from merlin.compiler.lower.interface import to_interface

    if source.kind == INTERFACE_IR:
        parse_interface(source.text)
        return source
    if source.kind != "mlir":
        raise ValueError("Interface preparation requires imported MLIR or explicit Merlin Interface IR")
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

    module = to_interface(parse_mlir_text(source.text))
    return Source(interface_text(module), kind=INTERFACE_IR, origin=source.origin)


class InterfaceTicket:
    """Carry selected-source custody across a shared, immutable IR conversion.

    A backend can acknowledge the original recipe and its actual artifacts while
    validating only Interface IR. It cannot substitute another upstream region.
    """

    def __init__(self, ticket, original, prepared):
        ticket.revalidate(original)
        self._ticket, self._original, self._prepared = ticket, original, prepared
        self._identity = request_identity(prepared)
        self.recipe = ticket.admitted.proposal.lowering
        self._recipe = self.recipe

    @property
    def sha256(self):
        return self._ticket.sha256

    def record(self):
        self.revalidate(self._prepared)
        return self._ticket.record()

    def revalidate(self, request):
        self._ticket.revalidate(self._original)
        if request_identity(request) != self._identity or self.recipe is not self._recipe:
            raise ValueError("selected Interface request or lowering recipe changed")

    def acknowledgment(self, request, result):
        self.revalidate(request)
        return self._ticket.acknowledgment(self._original, result)

    def admit(self, request, result):
        self.revalidate(request)
        self._ticket.admit(self._original, result)


@dataclass(frozen=True)
class InterfaceHandoff:
    original: CompileRequest
    request: CompileRequest
    selected: object | None
    _original_identity: tuple
    _prepared_identity: tuple
    _provider: object

    def revalidate(self):
        if (
            request_identity(self.original) != self._original_identity
            or request_identity(self.request) != self._prepared_identity
            or input_contract(self._provider, self.original) != INTERFACE_IR
        ):
            raise ValueError("original request, Interface request or selected backend contract changed")
        if self.selected is not None:
            self.selected.revalidate(self.request)

    def publish(self, result):
        """Expose the compiler boundary beside its products, under shared ownership."""
        self.revalidate()
        path = result.directory / "merlin.interface.mlir"
        if path.exists():
            raise ValueError("backend occupied the Merlin-owned Interface artifact path")
        path.write_text(self.request.source.text, encoding="utf-8")
        published = replace(
            result,
            source_sha256=self.original.source.sha256,
            artifacts=(*result.artifacts, Artifact.from_file(path, kind="mlir", role="compiler-interface")),
            metadata={
                **result.metadata,
                "compiler_boundary": {
                    "schema": "merlin.interface_handoff.v1",
                    "input_ir": INTERFACE_IR,
                    "source_sha256": self.original.source.sha256,
                    "interface_sha256": self.request.source.sha256,
                    "preparation": "merlin.compiler.lower.interface",
                    "semantics_verified": False,
                },
            },
        )
        if self.selected is not None:
            # Keep the provider's exact product acknowledgment, then record the
            # final shared roster including this Merlin-owned boundary artifact.
            published = replace(
                published,
                metadata={
                    **published.metadata,
                    "provider_selected_lowering": result.metadata.get("selected_lowering"),
                    "selected_lowering": self.selected.acknowledgment(self.request, published),
                },
            )
            self.selected.admit(self.request, published)
        return published


def prepare(request, provider, selected=None):
    """Prepare only explicitly declared providers; legacy providers are unchanged."""
    original = request_identity(request)
    contract = input_contract(provider, request)
    if request_identity(request) != original:
        raise ValueError("backend contract selection changed the compilation request")
    if contract is None:
        return None
    source = interface_source(request.source)
    # Staged preparation remains bound to the original source; its admission is
    # performed there before this handoff, never rebound to transformed bytes.
    prepared = CompileRequest(source, request.output, request.options)
    ticket = InterfaceTicket(selected, request, prepared) if selected is not None else None
    result = InterfaceHandoff(request, prepared, ticket, original, request_identity(prepared), provider)
    result.revalidate()
    return result
