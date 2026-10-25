"""Target capability admission for compute-group stages and declared readouts."""
from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

@dataclass
class Admission:
    """What the target says about one stage attached to a contraction."""

    admitted: bool
    refusal: str | None = None
    reason: str = ""
    units: tuple[str, ...] = ()


class TargetOracle:
    """The target's answers, asked through the same oracles the rest of the system uses."""

    def __init__(
        self,
        target: str,
        *,
        readout: Any | None = None,
        prohibited_roles=(),
        semantic_facts=None,
        routes: Sequence[Any] | None = None,
    ):
        from merlin.targetgen import eligibility as E
        from merlin.targetgen import readout_facet

        self.target = target
        # The target's declared non-readout stage routes (a bias seeded into the accumulator before
        # the contraction, say). A stage no readout applies is still carried by the group when one
        # of these routes composes it with a contraction.
        self.routes = tuple(routes) if routes is not None else tuple(readout_facet.epilogue_stage_routes(target))
        # The experiment's prohibited instruction roles: a standalone form that needs one is refused.
        self.prohibited_roles, self.semantic_facts = tuple(prohibited_roles), semantic_facts
        self._E = E
        self.cap_map = E.capability_map_for_target(target)
        self.undetermined = E.undetermined_families_for_target(target)
        self.providers = E.providers_for_target(target)
        self.readout = (
            readout if readout is not None else readout_facet.TargetReadout(tuple(readout_facet.for_target(target)))
        )

    def unit_for(
        self, in_dtype: str | None, weight_dtype: str | None, rank: int | None, providers: Sequence[str]
    ) -> str:
        """The unit the target's own router gives a contraction of these formats."""
        from merlin.targetgen import routing as R

        try:
            routed = R.route_plan(
                [
                    R.OpDemand(
                        op="matmul", in_fmt=in_dtype or "", weight_fmt=weight_dtype, rank=rank, family="contraction"
                    )
                ],
                self.target,
            )["results"]
            unit = getattr(routed[0], "unit", None) if routed else None
        except Exception:  # noqa: BLE001 -- an unroutable demand leaves the declared provider
            unit = None
        return unit or (providers[0] if providers else "accelerator")

    def standalone_admission(self, family: str) -> dict[str, Any]:
        """Is ``family`` admitted standalone under the prohibited roles (``capability_roles``)?"""
        from merlin.targetgen import capability_roles as CR
        from merlin.targetgen import compute_units as CU
        from merlin.targetgen.target_registry import load_contract

        units = list(CU.compute_units(load_contract(self.target)))
        return CR.family_admission(units, family, prohibited_roles=self.prohibited_roles,
                                   semantic_facts=self.semantic_facts)  # fmt: skip

    def absorbs(self, kind: str) -> Admission:
        """Does a readout, or a declared stage route, carry ``kind`` out of a contraction with it?"""
        # Resolve the existing public vocabulary and injection seam at call time.
        from .compute_groups import (
            ACCUMULATE_UNPROVED, READOUT_DOES_NOT_APPLY, RESIDUAL_ADD,
            declared_stage_name, readout_absorbs,
        )

        verdict = readout_absorbs(kind, self.readout, target=self.target)
        if not verdict.admitted and kind == RESIDUAL_ADD and getattr(self.readout, "accumulates", False):
            # The hardware CAN add the residual into the accumulator; whether that integer sum rounds as
            # the capture's float sum does is a proof this pass does not have, so the residual and what
            # follows it stay on the host, named as such rather than as a missing declaration.
            return Admission(
                False,
                ACCUMULATE_UNPROVED,
                f"{self.target!r} accumulates an operand on load, but no proof shows the integer residual "
                "sum equals the captured float sum after rounding",
            )
        if verdict.admitted or verdict.refusal != READOUT_DOES_NOT_APPLY:
            return verdict
        stage = declared_stage_name(kind)
        if any(
            getattr(route, "stage", None) == stage and getattr(route, "composed_with", None) == "contraction"
            for route in self.routes
        ):
            return Admission(True)
        return verdict

    def absorbs_sequence(self, kinds: Sequence[str], *, unit: str) -> Admission:
        """Require one physical readout for all stages; numeric checks follow."""
        from .group_composition import absorbs_sequence

        return absorbs_sequence(self, kinds, unit=unit)

    def ask(
        self,
        *,
        op: str,
        family: str,
        in_dtype: str | None,
        weight_dtype: str | None = None,
        rank: int | None = None,
        attached: bool,
        granularity: str | None = None,
        form: str | None = None,
    ) -> Admission:
        region = self._E.RegionDescriptor(
            op=op, family=family, in_dtype=in_dtype, weight_dtype=weight_dtype,
            rank=rank, scale_granularity=granularity,
            form=form,
        )
        verdict = self._E.is_eligible(
            region,
            self.cap_map,
            undetermined=self.undetermined,
            providers=self.providers,
            fused_with=("contraction",) if attached else None,
            readout=self.readout if granularity is not None else None,
        )
        return Admission(verdict.eligible, verdict.refusal, verdict.reason, verdict.units)
