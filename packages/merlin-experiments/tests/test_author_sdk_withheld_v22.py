"""The acceptance suite's withheld-module checks pass on a V22 author SDK and still fail on V21.

Inputs are prepared the way ``prepare_author_library`` prepares ``author-library/base-sdk`` (the same
shared source release owner, the same arm tool set); the sandbox argument payload is composed from the
arm bundle's own grants after common-substrate attachment, which is what the live sandbox mounts.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from merlin_experiments.phase1 import author_acceptance as AA
from merlin_experiments.phase1 import full_compiler_workspace as FCW

from merlin.common.paths import repo_root
from merlin.targetgen import tool_registry as TR
from merlin.targetgen.generate_bundles import _select_python_grants
from merlin.targetgen.tool_profile import COMMON_SUBSTRATE_V14, COMMON_SUBSTRATE_V15, attach_common_substrate
from merlin.targetgen.tool_profile_shared import prepare_shared_source_release

TOOLS = TR.ARM_TOOLS["merlin_rtlchecks"]
SUBSTRATES = {21: COMMON_SUBSTRATE_V14, 22: COMMON_SUBSTRATE_V15}
#: What the acceptance stage script records as ``refused``: prefixes plus module-level policy denials.
REFUSED = sorted({*FCW.DENIED_PREFIXES, *(m for e in FCW.MODULE_ACCESS for m in e.modules)})
ORACLE = "merlin.targetgen.rtl.mlc_bridge"


@pytest.fixture(scope="module", params=(21, 22))
def inputs(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(f"inputs-v{request.param}")
    sdk = root / "author-library" / "base-sdk"
    prepare_shared_source_release(
        sdk,
        python_source_root=repo_root() / "src",
        license_directory=repo_root(),
        tools=TOOLS,
        version=request.param,
        source_binding=True,
    )
    return request.param, root, sdk


def _context(inputs_root: Path, work: Path) -> AA.Context:
    ws = work / "workspace"
    (ws / "submission" / "compiler_library").mkdir(parents=True)
    staged = {"workspace": str(ws), "denied_prefixes": list(FCW.DENIED_PREFIXES), "refused": REFUSED}
    return AA.Context(
        repo_root(), inputs_root, work, staged, AA.author_roots(staged, inputs_root), {}, None, "medium", 3
    )


def _mounted_args(sdk: Path, version: int, path: Path) -> Path:
    """The ``--ro-bind`` payload for every SDK file the arm bundle grants (directories expanded)."""
    allowed = [{"path": p, "mode": "ro", "note": "tool"} for t in TOOLS for p in TR.spec(t).bundle_paths]
    bundle = dict(arm="merlin_assisted", tools=list(TOOLS), allowed=allowed, denied=[], host_inputs=[])
    _select_python_grants(bundle, sdk)
    attach_common_substrate(bundle, version=SUBSTRATES[version], python_source_root=sdk, host_index_bits=64)
    _select_python_grants(bundle, sdk)
    argv = []
    for row in bundle["allowed"]:
        grant = Path(row["path"])
        if not grant.is_relative_to(sdk) or not grant.exists():
            continue
        for leaf in sorted(grant.rglob("*.py")) if grant.is_dir() else [grant]:
            argv += ["--ro-bind", str(leaf), str(leaf)]
    path.write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    return path


def test_withheld_imports_fail_on_v21_and_pass_on_v22(inputs, tmp_path):
    version, root, sdk = inputs
    ctx = _context(root, tmp_path)
    assert str(sdk) in [str(r) for r in ctx.roots]
    check = AA.check_withheld(ctx)
    if version == 21:
        assert check.status == AA.FAIL, check
        assert {"merlin.kernels.cca", "merlin.kernels.action_catalog", ORACLE} <= set(check.detail["candidates"])
    else:
        assert check.status == AA.PASS, check.cause
        assert check.detail["candidates"] == []


def test_module_level_policy_denials_are_checked_not_only_package_prefixes(inputs, tmp_path):
    version, root, sdk = inputs
    if version != 21:
        pytest.skip("the historical release is the one that still carries the oracle module")
    ctx = _context(root, tmp_path)
    assert ORACLE not in AA._withheld_candidates(ctx.roots, ctx.staged["denied_prefixes"])  # prefixes alone miss it
    assert ORACLE in AA._withheld_candidates(ctx.roots, AA.withheld_names(ctx.staged))
    # A source file on the author path fails the check even when it does not import there.
    (tmp_path / "only-oracle" / "merlin/targetgen/rtl").mkdir(parents=True)
    (tmp_path / "only-oracle" / "merlin/targetgen/rtl/mlc_bridge.py").write_text("raise ImportError('masked')\n")
    lone = AA.Context(ctx.framework, None, ctx.work, ctx.staged, [tmp_path / "only-oracle", ctx.ws], {}, None, "m", 3)
    check = AA.check_withheld(lone)
    assert check.status == AA.FAIL and ORACLE in check.cause


def test_sandbox_args_audit_fails_on_v21_mounts_and_passes_on_v22(inputs, tmp_path):
    version, root, sdk = inputs
    names = AA.withheld_names(_context(root, tmp_path).staged)
    check = AA.audit_sandbox_args(_mounted_args(sdk, version, tmp_path / "bwrap.args"), names)
    if version == 21:
        assert check.status == AA.FAIL
        mounted = {row.split(" at ")[0] for row in check.detail["withheld_mounted"]}
        assert {"merlin.kernels.cca", "merlin.kernels.microkernel", "merlin.kernels.endpoints", ORACLE} <= mounted
    else:
        assert check.status == AA.PASS, check.cause


_RTL_TOOLING = r"""
import contextlib, importlib, importlib.util, io, os, pathlib
facts, out = sys.argv[1], {}
sdk = pathlib.Path(sys.path[0])
rtl = sorted(p.stem for p in (sdk / "merlin/targetgen/rtl").glob("*.py") if p.stem != "mlc_bridge")
for stem in rtl:
    try:
        importlib.import_module("merlin.targetgen.rtl." + stem)
        out["import:" + stem] = "ok"
    except BaseException as exc:
        out["import:" + stem] = type(exc).__name__
for gen in ("gen_isa_module", "gen_rtl_digest", "gen_numeric_facts"):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            code = importlib.import_module("merlin.targetgen.rtl." + gen).main(["--facts", facts, "--target", "phantom5"])
        out[gen] = [code, buf.getvalue()]
    except BaseException as exc:
        out[gen] = type(exc).__name__ + ": " + str(exc)[:200]
os.environ["MERLIN_RTL_FACTS"] = facts
from merlin.targetgen import rtl_backend
profile = rtl_backend.target_profile("phantom5")
out["target_profile"] = [profile.dim, list(profile.legal_opcodes or ()), sorted((profile.memory_map or {}).get("memories", {}))]
out["mlc_bridge_importable"] = importlib.util.find_spec("merlin.targetgen.rtl.mlc_bridge") is not None
_emit(out)
"""


def test_rtl_facts_and_rtl_profile_tooling_behave_identically_without_the_oracle_bridge(tmp_path_factory):
    """Withholding the bridge costs the author nothing it could use: in the author interpreter (no mlc,
    no MERLIN_MLC_DIR) the RTL generators, the facts reader and the structural profile give the same
    answers on V22 as on V21, where the bridge was still staged."""
    facts = repo_root() / "merlin/tests/fixtures/phantom_target/phantom5/contracts/rtl_facts/facts.json"
    results = {}
    for version in (21, 22):
        sdk = tmp_path_factory.mktemp(f"rtl-v{version}") / "base-sdk"
        prepare_shared_source_release(
            sdk,
            python_source_root=repo_root() / "src",
            license_directory=repo_root(),
            tools=TOOLS,
            version=version,
            source_binding=True,
        )
        proc, out = AA.run_child(_RTL_TOOLING, [sdk], (), facts, cwd=sdk.parent)
        assert out is not None, proc.stderr[-2000:]
        results[version] = out
    assert results[21].pop("mlc_bridge_importable") is True
    assert results[22].pop("mlc_bridge_importable") is False
    assert results[22] == results[21]
    assert results[22]["target_profile"][0] is not None  # the reviewed facts answered, not live discovery


def test_tool_descriptions_name_only_what_the_bundle_grants(inputs, tmp_path):
    """ALLOWED_MERLIN_TOOLS.md is rendered from the grant: a narrowed CCA spine never advertises the
    operator-only sources, and names the broker calls that replace them."""
    from merlin.targetgen.generate_bundles import _tool_blurbs

    version, _root, sdk = inputs
    allowed = [{"path": p, "mode": "ro", "note": "tool"} for t in TOOLS for p in TR.spec(t).bundle_paths]
    bundle = dict(arm="merlin_assisted", tools=list(TOOLS), allowed=allowed, denied=[], host_inputs=[])
    _select_python_grants(bundle, sdk)
    attach_common_substrate(bundle, version=SUBSTRATES[version], python_source_root=sdk, host_index_bits=64)
    _select_python_grants(bundle, sdk)
    text = _tool_blurbs(bundle)
    spine = text.split("### `cca_spine`", 1)[1].split("### ", 1)[0]
    spec = TR.spec("cca_spine")
    assert [path for path, _ in spec.parts] == list(spec.bundle_paths)  # every grant has its description
    if version == 21:
        assert spine.strip() == spec.blurb  # historical bundles keep the reviewed text
        return
    for path, part in spec.parts:
        assert (part in spine) == path.endswith("rtl_backend.py"), part
    broker = TR.spec(spec.brokered_alternative).broker
    assert all(f"`{call}`" in spine for call in broker.calls)
    assert all(f"`{name}`" in spine for _, name in broker.shims)
    assert "extract a CCA" not in spine and "microkernel" not in spine


def test_brokered_calls_are_exactly_what_the_broker_and_its_clients_serve():
    import importlib

    from merlin_experiments.phase1.brokers import cca as broker

    from merlin.targetgen import rtl_backend

    spec = TR.spec("cca_tools").broker
    unknown = broker._handle({"cmd": "no-such-call"})["error"]
    assert all(call in unknown for call in spec.calls)
    for module, _ in spec.shims:
        client = importlib.import_module(module)
        assert all(callable(getattr(client, call)) for call in spec.calls)
    assert callable(rtl_backend.target_profile) and callable(rtl_backend.derived_levers)


def test_cca_broker_answers_operator_side_and_its_client_imports_no_merlin():
    """The CCA calls are a host broker over the operator framework, never the author SDK."""
    import ast
    import importlib

    from merlin_experiments.phase1.brokers import cca as broker

    import merlin.kernels.cca_contract as contract

    assert Path(contract.__file__).is_relative_to(repo_root() / "src")
    reply = broker._handle({"cmd": "escalation_ladder", "axis": "tile_shape", "target": "rvv"})  # target-ok
    assert "error" not in reply and reply["axis"] == "tile_shape"
    client = TR.spec("cca_tools").broker
    assert client is not None and {name for _, name in client.shims} == {"cca_contract.py", "action_catalog.py"}
    for module, _staged in client.shims:
        tree = ast.parse(Path(importlib.import_module(module).__file__).read_bytes())
        names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        assert not [n for n in names if n == "merlin" or n.startswith(("merlin.", "merlin_experiments"))], names
