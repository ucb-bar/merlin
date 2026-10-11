"""Fixed search selections for normal corpus and isolated component policies.

Normal search retains the Phase 1 functional fork and its original evaluator.
The two routes keep their own scientific admission and feedback schemas.
"""

from .contracts import StageGateError

ORDINARY_KIND = "corpus-rtl-feedback"


class SearchInputs:
    def __init__(self, policy):
        from .component_workflow import ComponentOnlyPolicy
        from .ordinary_search_policy import OrdinarySearchPolicy

        if type(policy) not in (ComponentOnlyPolicy, OrdinarySearchPolicy):
            raise StageGateError("search requires its fixed admitted policy")
        self.policy = policy
        self.ordinary = type(policy) is OrdinarySearchPolicy

    def validate(self):
        return self.policy.validate_search_inputs() if self.ordinary else self.policy._validate()

    @property
    def corpus(self):
        return self.policy.search_corpus if self.ordinary else self.policy.component_corpus

    @property
    def baseline(self):
        return self.policy.search_baseline if self.ordinary else self.policy.baseline_admission.baseline

    @property
    def rtl(self):
        return self.policy.feedback_evaluator if self.ordinary else self.policy.component_rtl

    @property
    def analytical(self):
        return None if self.ordinary else self.policy.component_analytical

    @property
    def timing_kind(self):
        return ORDINARY_KIND if self.ordinary else "component-rtl-feedback"

    @property
    def kinds(self):
        return (ORDINARY_KIND,) if self.ordinary else ("component-analytical-feedback", self.timing_kind)

    def identities(self):
        self.validate()
        if self.rtl is None:
            raise StageGateError("search requires its original explicitly selected timing evaluator")
        return {
            "manifest_sha256": self.corpus.manifest_sha256,
            "corpus_sha256": self.corpus.capsules_sha256,
            "baseline_sha256": self.rtl.baseline_sha256,
            "certificate_sha256": self.rtl.certificate.sha256,
        }

    def validate_result(self, raw, revision, kind):
        from .corpus_feedback import validate_redacted_feedback

        roster = self.validate()
        if self.ordinary:
            if kind != ORDINARY_KIND:
                raise StageGateError("normal corpus search selected another evaluator")
            result = validate_redacted_feedback(raw)
            if (
                result["candidate_sha256"] != revision.source_sha256
                or result["tuning_corpus_sha256"] != self.corpus.capsules_sha256
                or result["certificate_sha256"] != self.rtl.certificate.sha256
            ):
                raise StageGateError("normal queued result differs from its Phase 1 source/corpus/engine selection")
            rows = result["cells"]
        else:
            from .component_workflow import validate_component_feedback

            result = validate_component_feedback(raw)
            provider = self.analytical
            expected = (
                ("calibrated_component_analytical", provider.implementation_sha256, provider.configuration_sha256)
                if kind == self.kinds[0]
                else ("certified_component_rtl", self.policy._rtl_configuration[0][1], self.rtl.certificate.sha256)
            )
            if (
                (result["tier"], result["provider_sha256"], result["configuration_sha256"]) != expected
                or result["candidate_sha256"] != revision.source_sha256
                or result["corpus_sha256"] != self.corpus.capsules_sha256
                or result["manifest_sha256"] != self.corpus.manifest_sha256
            ):
                raise StageGateError("component queued result differs from its admitted selection")
            rows = result["evidence"] if kind == self.kinds[0] else result["evidence"]["cells"]
        identities = [(row["family"], row["capsule"]) for row in rows]
        if len(identities) != len(roster) or set(identities) != roster:
            raise StageGateError("queued evaluation omitted its exact admitted member roster")
        return result
