"""Explicit diagnostic member selection on the original normal RTL evaluator.

This grants no scientific authority: selected calls retain the original source,
numerical comparison and certificate. Unselected obligations remain unmeasured,
and this kind is excluded from complete measured promotion and plateau logic.
"""

from dataclasses import dataclass
from pathlib import Path

from .contracts import StageGateError, document_sha256, sha256_file

KIND = "corpus-selected-rtl-feedback"
ACTION = "evaluation-submit-selected"
_ISSUER = object()


def select_members(policy, members):
    roster = policy.validate_search_inputs()
    if type(members) is not tuple or not members or len(members) > len(roster):
        raise StageGateError("selected evaluation needs explicit nonempty public member identities")
    if any(type(key) is not tuple or len(key) != 2 or any(type(value) is not str for value in key) for key in members):
        raise StageGateError("selected evaluation requires the original family/capsule identity pairs")
    if len(set(members)) != len(members) or not set(members) <= roster:
        raise StageGateError("selected evaluation repeats or substitutes original public members")
    return tuple(sorted(members))


def parse_members(policy, text):
    if type(text) is not str or not text or len(text) > 8192:
        raise StageGateError("selected evaluation needs bounded comma-separated public family/capsule identities")
    roster = policy.validate_search_inputs()
    lookup = {family + "/" + capsule: (family, capsule) for family, capsule in roster}
    if len(lookup) != len(roster):
        raise StageGateError("selected evaluation public names are ambiguous")
    parts = text.split(",")
    if any(value not in lookup for value in parts):
        raise StageGateError("selected evaluation requested an unknown or ambiguous public identity")
    return select_members(policy, tuple(lookup[value] for value in parts))


@dataclass(frozen=True, eq=False)
class OrdinaryMeasurementPlan:
    policy: object
    evaluator: object
    revision: object
    baseline: object
    members: tuple
    selected: tuple
    member_timeout_s: int
    source_sha256: str
    _issuer: object

    def verify(self, evaluator, *, candidate=None, package=None, arm=None, member=None):
        self.policy.validate_search_inputs()
        if (
            self._issuer is not _ISSUER
            or evaluator is not self.evaluator
            or sha256_file(Path(__file__)) != self.source_sha256
        ):
            raise StageGateError("selected normal RTL plan changed its original execution owner")
        self.revision.verify(authority=self.policy.edit_authority)
        self.baseline.verify(authority=self.policy.edit_authority)
        if (
            evaluator.corpus is not self.policy.search_corpus
            or evaluator.certificate != self.policy.feedback_evaluator.certificate
            or evaluator.baseline != self.baseline.source
            or evaluator.baseline_sha256 != self.baseline.source_sha256
            or select_members(self.policy, self.members) != self.members
            or tuple(document_sha256(list(key)) for key in self.members) != self.selected
        ):
            raise StageGateError("selected normal RTL changed source, corpus, certificate or public scope")
        if candidate is not None and Path(candidate) != self.revision.source:
            raise StageGateError("selected normal RTL candidate differs from its issued retained source")
        if package is not None and (
            arm not in ("baseline", "candidate")
            or Path(package) != (self.baseline.source if arm == "baseline" else self.revision.source)
        ):
            raise StageGateError("selected normal RTL arm differs from its original retained source")
        if member is not None and (member.family, member.capsule) not in self.members:
            raise StageGateError("selected normal RTL attempted an unselected member")

    def reason(self, member):
        return "unselected public member; diagnostic RTL selection cannot establish complete objective or promotion"


def prepare(policy, evaluator, revision, baseline, *, members, timeout_s):
    from .development_feedback import DevelopmentGsimFeedback
    from .ordinary_search_policy import OrdinarySearchPolicy

    if type(policy) is not OrdinarySearchPolicy or type(evaluator) is not DevelopmentGsimFeedback:
        raise StageGateError("selected RTL requires its original normal admitted evaluator")
    members = select_members(policy, members)
    value = OrdinaryMeasurementPlan(
        policy,
        evaluator,
        revision,
        baseline,
        members,
        tuple(document_sha256(list(key)) for key in members),
        timeout_s,
        sha256_file(Path(__file__)),
        _ISSUER,
    )
    value.verify(evaluator, candidate=revision.source)
    return value


def validate_feedback(document, *, members):
    allowed = set(members)
    for row in document["cells"]:
        if (row["family"], row["capsule"]) not in allowed and (row["measured"] or row["comparable"]):
            raise StageGateError("selected RTL feedback measured an unselected original obligation")
    if document["stopping"]["status"] != "undeterminable":
        raise StageGateError("selected RTL feedback cannot establish convergence")
    return document
