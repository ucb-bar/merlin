"""Select author compiler tools separately from operator simulation resources.

The target descriptor's simulator adapter owns its resources. A verified author
information policy decides whether those resources belong in an author/candidate
namespace. Operator execution keeps using sandbox.toolchain directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from merlin.common.text_tokens import is_identifier
from merlin.targetgen.sandbox import toolchain as TC


@dataclass(frozen=True)
class AuthorToolchainSelection:
    """A profile-bound role selection, not target or simulator qualification."""

    profile_name: str | None
    simulator_support: bool
    withheld_roots: tuple[Path, ...] = ()
    withheld_environment: tuple[str, ...] = ()

    def record(self) -> dict:
        return dict(
            schema="merlin.author_toolchain_selection.v1",
            profile=self.profile_name,
            roles=["compiler", *(["simulation_support"] if self.simulator_support else [])],
            withheld_roots=[str(path) for path in self.withheld_roots],
            withheld_environment=list(self.withheld_environment),
        )


def select(target, bundle: dict) -> AuthorToolchainSelection:
    """Use exact public profile disclosure; legacy/unspecified policy is unchanged.

    Simulator directories may contain derived hardware descriptions as well as
    executables. When extracted target facts are withheld, that entire resource
    role (including a declared simulation harness) stays with the host judge.
    No target name, simulator spelling or filesystem layout selects permission.
    """
    record = bundle.get("author_profile")
    if record is None:
        return AuthorToolchainSelection(None, True)
    if not isinstance(record, dict):
        raise ValueError("author toolchain requires an original author profile record")
    from merlin.development.backend import author_profiles as AP

    profile = AP.resolve(record.get("name"))
    if profile is None:
        raise ValueError("author toolchain requires an original author profile selection")
    AP.verify_bundle(profile, bundle)
    disclosure = getattr(profile, "disclosure", None)
    if disclosure is None or disclosure.get("rtl_facts") is not False:
        return AuthorToolchainSelection(profile.name, True)
    roots = tuple(AP.withheld_simulator_roots(target, bundle))
    # The selected runtime adapter declares these variables. Leaving them in a
    # child environment would advertise a resource that the profile withholds.
    variables = set(TC._sim(target).env_extra)
    if TC.curated_harness_dir(target):
        variables.update(("MERLIN_HWBRINGUP_HARNESS_DIR", f"MERLIN_{target.target.upper()}_HARNESS_DIR"))
    if any(not isinstance(variable, str) or not is_identifier(variable) for variable in variables):
        raise ValueError("simulator adapter environment names must be shell identifiers")
    return AuthorToolchainSelection(profile.name, False, roots, tuple(sorted(variables)))


def toolchain_binds(target, bundle: dict) -> list[str]:
    selected = select(target, bundle)
    if selected.simulator_support:
        return TC.toolchain_binds(target)
    return [
        *TC.toolchain_binds(target, sim=TC.SimToolchain(), harness=""),
        *(value for name in selected.withheld_environment for value in ("--unsetenv", name)),
    ]


def sandbox_env(target, workspace: Path, *, bundle: dict) -> str:
    selected = select(target, bundle)
    if selected.simulator_support:
        return TC.sandbox_env(target, workspace, bundle=bundle)
    environment = TC.sandbox_env(target, workspace, bundle=bundle, sim=TC.SimToolchain(), harness="")
    return environment + "".join(f"unset {name}; " for name in selected.withheld_environment)


def required_tool_probes(target, *, bundle: dict) -> list[TC.ToolProbe]:
    selected = select(target, bundle)
    if selected.simulator_support:
        return TC.required_tool_probes(target)
    return TC.required_tool_probes(target, sim=TC.SimToolchain())
