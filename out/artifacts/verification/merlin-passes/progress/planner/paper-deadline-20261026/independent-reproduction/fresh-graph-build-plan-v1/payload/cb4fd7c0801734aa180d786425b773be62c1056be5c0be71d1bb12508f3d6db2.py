"""Separate SOURCE-ONLY larger graph profiles. No compiler authority or SDK."""
from dataclasses import dataclass

SUFFIX = ("scalar_redundancy_elimination", "scalar_loop_invariant_motion",
          "ordered_loop_unrolling", "ordered_loop_segmentation")


@dataclass(frozen=True)
class Profile:
    name: str
    kinds: tuple[str, ...]
    domain: str
    relation: str
    premises: tuple[str, ...]
    final_root: str


COMMON = (
    "trusted complete xDSL parsing, actual source projection and host source binding",
    "same Scalar.Program/evalProgram, concrete mathematical Int index control and complete terminal world",
    "external calls explicitly declared; same possibly failing descriptor interpretation with local successful arity",
    "no internal call bridge, native ABI/alias/allocator/completion fidelity or hidden private-SSA observers",
    "one static nonnegative changed top-level loop per changed function, positive stride, no-tail bulk count>factor",
    "factor2..4, tile2..16, signed32|64 control bounds, one..four word carries, two-input changed functions",
    "region-free original loop body, at most16 lane external calls, bounded syntax, no broader control modes",
)
SCF = Profile("scalar_scf_four_transform_source_v1", SUFFIX,
              "complete_scalar_scf_static_no_tail_entries_v1",
              "equal_all_entry_returns_errors_terminal_world",
              COMMON + ("one total deterministic state-free primitive interpretation; no source-success premise",),
              "merlin_profile_transform")
INTEGER = Profile("canonical_integer_scalar_scf_source_v1", ("exact_integer_simplification", *SUFFIX),
                  "canonical_word_entry_scalar_scf_static_no_tail_v1",
                  "equal_canonical_entry_returns_errors_terminal_world",
                  COMMON + ("existing wrapping BitVec word primitive; canonical signless word entry values",
                            "local external successful-result canonicality on canonical operands; reached typing derived",
                            "wrong arity and absent entry remain observable; no assumed source success or intermediate typing"),
                  "merlin_profile_transform")


def selected(name):
    for profile in (SCF, INTEGER):
        if type(name) is str and name == profile.name:
            return profile
    raise ValueError("explicit larger graph profile required; no minimal/default/legacy fallback")


@dataclass(frozen=True)
class Settings:
    factor: int
    tile: int
    index_bits: int

    def validate(self):
        if (any(type(v) is not int for v in (self.factor, self.tile, self.index_bits))
                or not 2 <= self.factor <= 4 or not 2 <= self.tile <= 16 or self.index_bits not in (32, 64)):
            raise ValueError("exact factor2..4/tile2..16/index32|64 settings required")


def public_check(*args, **kwargs):
    raise RuntimeError("larger graph profiles UNSUPPORTED/HELD; no reviewed source/native activation")


def compiler_eligible(*args, **kwargs):
    # No process-local authority ledger is issued by a preparation adapter.
    return False
