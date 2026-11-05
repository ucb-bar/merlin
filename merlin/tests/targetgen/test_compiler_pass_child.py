"""Fresh original-parent children rerun pass admission under sealed transport."""

import importlib.util
import json
import os

import pytest
from merlin_experiments.phase1.feedback import compiler_profile_child as CHILD
from test_compiler_profile_child import _CONTROL, _run, _scope
from test_compiler_profile_child import original as original

from merlin.llvmlower.reviewed_pass_capabilities import PassCapabilityBinding
from merlin.targetgen.compiler_pass_selection import (
    CompilerPassSelection,
    OriginalPassFactory,
    active_compiler_passes,
    scoped_compiler_passes,
)

FACTORY = """from pathlib import Path
from merlin.common.digest import sha256_file
from merlin.llvmlower.reviewed_pass_capabilities import (
 PassCapabilityBinding,ReviewedPassCapabilities,ReviewedPassCapability)
class Owner:
 def __init__(self,path):
  self.path=Path(path);self.identity=sha256_file(self.path)
 def revalidate(self):
  if sha256_file(self.path)!=self.identity: raise ValueError('original evidence changed')
def admit(inputs, *, work_root):
 path=Path(inputs['evidence'])
 if path.read_text()!='original neutral admission': raise ValueError('independent admission refused')
 work_root.mkdir(parents=True,exist_ok=False)
 (work_root/'actual-independent-check').write_text(sha256_file(path))
 binding=PassCapabilityBinding('rtl','sw','numeric','runtime','library')
 return ReviewedPassCapabilities((ReviewedPassCapability('configuration_cleanup',binding,Owner(path)),))
"""


def selected(original):
    path = original.private / "original_factory.py"
    path.write_text(FACTORY)
    spec = importlib.util.spec_from_file_location("selected_original_pass_factory", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evidence = original.private / "pass-evidence"
    evidence.write_text("original neutral admission")
    factory = OriginalPassFactory(module.admit, {"evidence": str(evidence)})
    return CompilerPassSelection(
        factory.create(work_root=original.private / "initial-pass-admission"),
        PassCapabilityBinding("rtl", "sw", "numeric", "runtime", "library"),
        mechanisms=("configuration_cleanup",),
        factory=factory,
    )


def test_real_child_replays_independent_admission_and_preserves_original_owner(original):
    passes = selected(original)
    script = _CONTROL.replace(
        "    job=json.loads(raw)",
        """    job=json.loads(raw)
    from merlin.targetgen.compiler_pass_selection import active_compiler_passes
    selected=active_compiler_passes()
    assert selected.resolve('configuration_cleanup') is not None
    assert receipt['schema']==CHILD.PASSES_SCHEMA
    assert receipt['compiler_passes_sha256']==CHILD._digest(selected.record())
    assert (spec.parent/'compiler/pass-admission/actual-independent-check').is_file()""",
    )
    with scoped_compiler_passes(passes), _scope(original):
        with CHILD.prepare_model_child(
            spec_path=original.spec, environment=dict(os.environ), package_dir=original.candidate
        ) as transport:
            assert transport.document["schema"] == CHILD.PASSES_SCHEMA
            result = _run(transport, original, script=script)
            assert result.returncode == 0, result.stderr
            transport.verify_result(json.loads(result.stdout))
            assert active_compiler_passes() is passes
    assert active_compiler_passes() is None


@pytest.mark.parametrize("damage", ["source", "evidence", "legacy", "missing", "binding"])
def test_stale_missing_or_downgraded_pass_selection_refuses_actual_child(original, damage):
    passes = selected(original)
    with scoped_compiler_passes(passes), _scope(original):
        with CHILD.prepare_model_child(
            spec_path=original.spec, environment=dict(os.environ), package_dir=original.candidate
        ) as transport:
            if damage in {"source", "evidence"}:
                path = original.private / ("original_factory.py" if damage == "source" else "pass-evidence")
                # Restore before the parent closes its still-live scope; the
                # child must independently detect the actual intervening drift.
                before = path.read_bytes()
                path.write_text("changed original selection")
                try:
                    result = _run(transport, original)
                    assert result.returncode != 0
                finally:
                    path.write_bytes(before)
            else:
                document = {**transport.document}
                if damage == "legacy":
                    document["schema"] = CHILD.SCHEMA
                elif damage == "missing":
                    del document["compiler_passes"]
                else:
                    document = json.loads(json.dumps(document))
                    document["compiler_passes"]["binding"]["runtime"] = "foreign-runtime"
                with CHILD._sealed_transport(document, executor=transport.executor, boundary=transport.boundary) as bad:
                    result = _run(bad, original)
                    assert result.returncode != 0
