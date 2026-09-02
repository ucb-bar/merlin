"""Gates that keep the synthesis loop honest once it is closed.

Three failure modes, each of which would let a corpus look complete while proving less than it claims:
a cell "covered" by growing the ratchet rather than by a capsule; a composition shape credited to a
grammar that structurally cannot express it; and a synthesis whose output drifts between runs, which
would make every downstream diff unreadable.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import yaml

from merlin.common.paths import merlin_dir, repo_root
from merlin.targetgen import boundary as B
from merlin.targetgen import corpus_synth as CS


def _specs() -> list[str]:
    root = merlin_dir() / "contract/capsules/conformance"
    return sorted(p.stem for p in root.glob("*.yaml")) if root.is_dir() else []


def _spec(target: str) -> dict:
    return yaml.safe_load(
        (merlin_dir() / "contract/capsules/conformance" / f"{target}.yaml").read_text(
            encoding="utf-8")) or {}


class TestGrammarCannotClaimAShapeItCannotExpress:
    """The `merlin_iface` grammar carries no host computation -- its whole op set is accelerator work.

    So a capsule in that grammar is `A` or `A->A` and can never be `H->A->H`, `A->H->A`, `routing` or
    `H`. That invariant is what makes the composition axis meaningful, and it lives in a parser that a
    future edit could quietly break: adding one host-ish mnemonic would let an iface capsule claim a
    seam it does not contain, and the corpus would report the shape covered.
    """

    _HOST_SHAPES = (B.H_A_H, B.A_H_A, B.ROUTING, B.HOST_ONLY)

    def test_an_iface_capsule_can_only_ever_be_A_or_A_to_A(self):
        text = (
            'module attributes {merlin_iface.version = "0.1", merlin_iface.target = "t",\n'
            '                   merlin_iface.abi_version = "0.1"} {\n'
            '  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<16x16xi8>\n'
            '  %A0 = merlin_iface.tensor {name = "A0", role = "input"} : tensor<16x16xi8>\n'
            '  %R = merlin_iface.resident_pack %W {layout = "packed_rhs"} :'
            ' (tensor<16x16xi8>) -> !merlin_iface.resident\n'
            '  %acc = merlin_iface.matmul %A0, %R :'
            ' (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>\n'
            '  %Y0 = merlin_iface.commit %acc {name = "Y0", epilogue = [], output_dtype = "i32"} :'
            ' (!merlin_iface.acc<i32>) -> tensor<16x16xi32>\n}\n')
        prof = B.profile_iface_text(text)
        assert prof.kind in (B.A, B.A_A, B.UNKNOWN), prof.kind
        assert not (set(prof.contains) & set(self._HOST_SHAPES)), (
            "the iface grammar carries no host computation, so it cannot contain a host seam")

    @pytest.mark.parametrize("target", _specs())
    def test_no_shipped_iface_capsule_claims_a_host_shape(self, target):
        """The same invariant, over what actually ships rather than a fixture."""
        from merlin.targetgen.target_experiment import load_target_experiment

        desc = merlin_dir() / "experiments/capsule_bench/targets" / target / "target_experiment.yaml"
        if not desc.is_file():
            pytest.skip(f"no descriptor for {target}")
        te = load_target_experiment(desc)
        for root in te.graded_roots():
            for cy in sorted(root.glob("*/capsule.yaml")):
                prof = B.profile_capsule(cy.parent, target)
                if prof.grammar != "merlin_iface":
                    continue
                claimed = set(prof.contains) & set(self._HOST_SHAPES)
                assert not claimed, (
                    f"{cy.parent.name} is a merlin_iface capsule claiming host shape(s) "
                    f"{sorted(claimed)}, which that grammar cannot express")


class TestSynthesisIsReproducible:
    @pytest.mark.parametrize("target", _specs())
    def test_the_tracked_file_matches_a_fresh_derivation(self, target):
        """`--check` is the cheap half of byte-stability: it needs neither torch nor an oracle, so it
        can run anywhere, and it catches a hand-edit to a file whose whole point is that it is
        evidence rather than authorship."""
        tracked = merlin_dir() / "contract/capsules/profiles" / f"{target}.synth.yaml"
        if not tracked.is_file():
            pytest.skip(f"{target} has no synthesized entries yet")
        proc = subprocess.run(
            [sys.executable, str(repo_root() / "build_tools/scripts/synth_capsule_corpus.py"),
             "--target", target, "--check"],
            capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, f"{target} synth file has drifted:\n{proc.stdout}{proc.stderr}"

    @pytest.mark.parametrize("target", _specs())
    def test_two_derivations_agree(self, target):
        assert CS.synthesize(_spec(target)) == CS.synthesize(_spec(target))


class TestCoverageMustBeEarned:
    @pytest.mark.parametrize("target", _specs())
    def test_a_synthesized_capsule_names_the_cell_it_covers(self, target):
        """A cell is covered because a capsule was built for it, not because a ratchet line was added.
        The provenance is what lets a reviewer tell those apart -- an entry that does not name its cell
        could be covering anything."""
        tracked = merlin_dir() / "contract/capsules/profiles" / f"{target}.synth.yaml"
        if not tracked.is_file():
            pytest.skip(f"{target} has no synthesized entries yet")
        doc = yaml.safe_load(tracked.read_text(encoding="utf-8")) or {}
        required = {c["cell"] for c in (_spec(target).get("cells") or ())}
        named = set()
        for entry in doc.get("capsules") or ():
            assert entry["source_role"] == CS.SOURCE_ROLE
            for cell in required:
                if cell in entry.get("source_reference", ""):
                    named.add(cell)
        # ...or NAMED AS UNWRITABLE. A required cell whose op has no direct-MLIR builder at a dtype the
        # PyTorch writer cannot express has no entry by construction, and the honest handling is to
        # report it as an uncovered cell rather than to leave it silently absent from both lists.
        reported = " ".join((doc.get("provenance") or {}).get("cells_no_writer_can_express") or ())
        unaccounted = {c for c in required - named if c not in reported}
        assert not unaccounted, f"cells neither covered nor reported unwritable: {unaccounted}"
