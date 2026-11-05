"""Synchronous execution of native mixed-region products with source-owned views.

This is generic launch/buffer composition, not target instruction lowering.
Callbacks remain responsible for target execution and completion qualification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from merlin.verify.lean_contract import document_digest


@dataclass(frozen=True)
class ConnectedInvocation:
    """One original SSA stage, with arguments in source operand/result order."""

    stage: object
    product: object
    bindings: object
    images: tuple[bytes, ...] | None


@dataclass(frozen=True)
class ConnectedProgram:
    region: object
    invocations: tuple[ConnectedInvocation, ...]
    continuation: object
    selected_contracts: object
    windows: tuple
    byte_orders: tuple[str, ...]
    checked_identity: str
    report: dict

    def _identity(self):
        return document_digest(
            {
                "continuation": asdict(self.continuation),
                "selection": self.selected_contracts.record(),
                "windows": [asdict(w) for w in self.windows],
                "byte_orders": self.byte_orders,
                "invocations": [
                    {
                        "bindings": asdict(i.bindings),
                        "ordinal": i.stage.ordinal,
                        "dense": {
                            "schedule": i.product.compiled.kernel.digest(),
                            "selection": i.product.selection_identity,
                        }
                        if i.images is None
                        else None,
                        "images": None if i.images is None else [b.hex() for b in i.images],
                    }
                    for i in self.invocations
                ],
            }
        )

    @classmethod
    def checked(cls, region, products, windows, byte_orders, selected_contracts, continuation):
        from merlin.compile.scheduling.pointwise_source import PointwiseBindings

        invocations = []
        for stage, product in zip(region.stages, products, strict=True):
            bindings = PointwiseBindings(
                tuple(windows[i] for i in stage.inputs),
                windows[stage.output],
                tuple(byte_orders[i] for i in (*stage.inputs, stage.output)),
                True,
            )
            images = None if stage.kind == "contraction" else product.images()
            invocations.append(ConnectedInvocation(stage, product, bindings, images))
        launches = sum(1 if i.images is None else len(i.images) for i in invocations)
        if launches > continuation.max_segments:
            raise ValueError("connected executable exceeds runtime launch budget")
        result = cls(
            region,
            tuple(invocations),
            continuation,
            selected_contracts,
            windows,
            byte_orders,
            "",
            {
                "source": region.record(),
                "bindings": [asdict(w) for w in windows],
                "runtime_continuation": asdict(continuation),
                "launches": launches,
                "stage_forms": [
                    "checked_compact_dense" if i.images is None else "checked_pointwise_images" for i in invocations
                ],
                "maturity": "executable_unfused_composition_not_cross_stage_fusion",
                "residency": "selected_runtime_buffer_persistence_only",
                "execution_qualification": "not_established",
                "physical_savings_claimed": False,
            },
        )
        object.__setattr__(result, "checked_identity", result._identity())
        result.revalidate()
        return result

    def revalidate(self):
        from merlin.compile.scheduling.dense_source import DenseSourceResult
        from merlin.compile.scheduling.pointwise_source import PointwiseCompilation

        self.region.revalidate()
        self.selected_contracts.revalidate()
        if self._identity() != self.checked_identity or len(self.invocations) != len(self.region.stages):
            raise ValueError("connected runtime bindings or continuation changed; recompile")
        for stage, invocation in zip(self.region.stages, self.invocations, strict=True):
            product = invocation.product
            if (
                invocation.stage != stage
                or product.source is not self.region.source
                or product.operation is not stage.operation
            ):
                raise ValueError("connected native product no longer covers its original stage")
            if stage.kind == "contraction":
                if type(product) is not DenseSourceResult or product.operations != (stage.operation,):
                    raise ValueError("connected dense stage must preserve each separate source operation")
                product.revalidate()
            elif type(product) is not PointwiseCompilation or product.images() != invocation.images:
                raise ValueError("connected pointwise image changed; recompile")

    def ordered_invocations(self):
        """Checked native products/bindings for ordinary guest code generation.

        Dense products expose their existing emit_c/emit_llvm APIs; pointwise
        entries carry source-checked images. The caller supplies target pointer
        ABI, image loading and synchronous launch emission. Returning this plan
        does not establish a compiled guest ABI or runtime qualification.
        """
        self.revalidate()
        return self.invocations

    def execute(self, backings, launch, *, backing_origins=None):
        """Run checked stages over explicit persistent bytearrays, returning live-outs.

        ``launch(invocation, inputs, output)`` consumes the native dense product
        or checked pointwise images. Inputs are read-only byte views; output is
        the exact writable SSA result view. Return None only after completion and
        visibility. Callback/runtime qualification is external, not established
        by this Python interface. No scheduling, packing or arithmetic is supplied.

        Backings use selected physical storage identities. Distinct identities
        must have distinct bytearrays; implicit aliasing through arbitrary buffer
        exporters is intentionally unsupported. Non-overlapping windows of one
        backing remain legal. Caller owns allocation and initialization.
        ``backing_origins`` maps every physical backing to the byte address
        represented by buffer[0]. Omission explicitly selects origin zero; a
        physical address is never rebased implicitly to fit a smaller buffer.
        """
        if type(backings) is not dict or not callable(launch):
            raise ValueError("connected execution requires explicit backing buffers and synchronous launcher")
        self.revalidate()
        external = {w.resource: w for w in self.selected_contracts.storage.external}
        required = {external[w.space].backing for w in self.windows}
        if any(name not in backings or type(backings[name]) is not bytearray for name in required):
            raise ValueError("connected execution requires an explicit bytearray per physical backing")
        if len({id(backings[name]) for name in required}) != len(required):
            raise ValueError("distinct selected backing identities alias at runtime")
        origins = {name: 0 for name in required} if backing_origins is None else backing_origins
        if (
            type(origins) is not dict
            or set(origins) != required
            or any(type(origin) is not int or origin < 0 for origin in origins.values())
        ):
            raise ValueError("runtime backing origins must explicitly cover every backing in byte-address units")
        views = []
        for window in self.windows:
            name = external[window.space].backing
            buffer = backings[name]
            start = window.start - origins[name]
            if start < 0 or len(buffer) < start + window.size:
                raise ValueError("connected runtime backing is smaller than its selected tensor window")
            views.append(memoryview(buffer)[start : start + window.size])
        for invocation in self.invocations:
            self.revalidate()
            inputs = tuple(views[i].toreadonly() for i in invocation.stage.inputs)
            output = views[invocation.stage.output]
            if launch(invocation, inputs, output) is not None:
                raise ValueError("connected launcher must complete synchronously and return None")
            self.revalidate()
        return {i: views[i].toreadonly() for i in self.region.outputs}
