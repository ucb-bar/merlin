"""Public cross-target comparison from explicitly selected experiment definitions."""

from pathlib import Path

from merlin_experiments.spec import SpecError

from .declarations import from_definition
from .profiles import compare_public_profiles, load_profile


def build(definitions: list[Path]) -> dict:
    declarations = [from_definition(path) for path in definitions]
    targets = [item.target for item in declarations]
    if len(targets) < 2 or len(set(targets)) != len(targets):
        raise SpecError("comparison requires at least two definitions with distinct targets")
    profiles = []
    for item in declarations:
        try:
            profile = load_profile(item.profile, include_holdouts=False, **item.profile_inputs())
        except ValueError as exc:
            raise SpecError(f"invalid public comparison input: {exc}") from exc
        profiles.append((item.target, profile))
    result = compare_public_profiles(profiles)
    result["scope"] = "current-declared-public-recipes"
    result["definitions"] = [str(item.definition) for item in declarations]
    return result
