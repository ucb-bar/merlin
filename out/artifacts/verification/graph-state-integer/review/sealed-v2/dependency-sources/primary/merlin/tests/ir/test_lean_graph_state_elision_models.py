"""Bound concrete Lean and native fixture models; explicit admission counterexample."""

import json
import os
import shutil
import subprocess
import sys

import pytest
from test_lean_graph_state_elision import CASES, REFERENCE, produce

from merlin.common.paths import repo_root
from merlin.verify import lean_graph_state as state
from merlin.verify.lean_graph_state_elision_kernel import check_query
from merlin.verify.lean_graph_state_elision_model import emit_witness, operands


@pytest.mark.parametrize("case,outcome", [("basic", "actual_success"), ("unknown-read", "actual_failure")])
def test_actual_complete_roster_initialized_model(case, outcome, tmp_path):
    _, _, _, projection = produce(tmp_path, CASES[case])
    names = ["model_admitted", "joint_model", outcome]
    result = check_query(
        emit_witness(projection),
        tmp_path / "model",
        roots=["Merlin.GraphStateElisionInstance." + name for name in names],
    )
    assert result["status"] == "conditional_model_checked", result
    assert result["audit"]["raw"]["violations"] == []
    assert not result["source_success_proved"]  # Only this concrete fixture model succeeds.


def diagnostic(root, program):
    path = root / "diagnostic.py"
    path.write_text(program)
    process = subprocess.run(
        [sys.executable, str(path)],
        cwd=REFERENCE,
        env={**os.environ, "PYTHONPATH": str(REFERENCE / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    (root / "diagnostic.stdout").write_text(process.stdout)
    (root / "diagnostic.stderr").write_text(process.stderr)
    state._write(
        root / "execution.json",
        {"returncode": process.returncode, "program": state.file_ref(path), "python": state.file_ref(sys.executable)},
    )
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout)


def test_signed_zero_kernel_comparator_counterexample_not_text_coverage(tmp_path):
    script = repo_root() / "src/merlin/verify/lean_graph_state_elision_capture.py"
    result = diagnostic(
        tmp_path,
        f"""
import importlib.util,json
from merlin.sched.ir import Call,Const,Kernel
from merlin.compile.scheduling.configuration import ConfigurationAdapter,verify_configuration_rewrite
s=importlib.util.spec_from_file_location('owned_projection',{str(script)!r})
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
a=Kernel('commands',(),(Call('observe',(('value',Const(-0.0)),)),))
b=Kernel('commands',(),(Call('observe',(('value',Const(0.0)),)),))
r=verify_configuration_rewrite(a,b,ConfigurationAdapter(lambda c:None,'fixture','provisional'))
print(json.dumps({{'production_accepts':bool(r),'exact_original':m.exact(a),'exact_changed':m.exact(b),
                  'independent_exact_equal':m.exact(a)==m.exact(b),'text_domain_supported':False}}))
""",
    )
    assert result["production_accepts"] and not result["independent_exact_equal"]
    assert result["text_domain_supported"] is False


def test_false_pure_hidden_pulse_counterexample(tmp_path):
    _, _, _, projection = produce(tmp_path)

    def pulses(atoms):
        # Alternative device violates exact whole-state setter admission.
        return sum(operands(atom)[0] == "setup" for atom in atoms)

    evidence = {
        "source_pulses": pulses(projection["before"]),
        "target_pulses": pulses(projection["after"]),
        "admission_satisfied": False,
        "matching_registers_suffice": False,
        "interpretation": "every setup additionally increments an observable pulse counter",
    }
    state._write(tmp_path / "counterexample.json", evidence)
    assert evidence["source_pulses"] == 2 and evidence["target_pulses"] == 1


def test_actual_hoist_cleanup_unavailable_same_kernel(tmp_path):
    result = diagnostic(
        tmp_path,
        """
import json
from merlin.sched.ir import parse_kernel
from merlin.compile.scheduling.configuration import (
 ConfigurationAdapter,ConfigurationEffects,eliminate_redundant_configuration)
from merlin.compile.scheduling.configuration_loops import hoist_loop_configuration
def decode(call):
 if call.instr=='setup':
  return ConfigurationEffects(writes=(('mode',call.arg('value').value),),ordered=True,pure_write=True)
 return ConfigurationEffects(reads=(('mode',7),),ordered=True)
adapter=ConfigurationAdapter(decode,'fixture','PROVISIONAL')
source=parse_kernel('kernel commands()\\n  for i in 0..3:\\n    setup(value=7)\\n    work()\\n')
middle=hoist_loop_configuration(source,adapter).kernel
try:
 final=eliminate_redundant_configuration(middle,adapter).kernel
 result={'status':'unexpectedly_supported','final':final.text()}
except ValueError as error:
 result={'status':'unavailable','reason':str(error),'source':source.text(),'actual_middle':middle.text(),'genuine_hoist':source!=middle}
print(json.dumps(result))
""",
    )
    assert result["status"] == "unavailable" and result["genuine_hoist"]
    assert "straight" in result["reason"] or "loop" in result["reason"].lower()


NATIVE = r"""
#include <stdio.h>
#include <stdint.h>
int main(void) {
  int op; long long a,b,mode=-13,width=0,obs=2,pulses=0,status=0;
  while (scanf("%d %lld %lld", &op,&a,&b)==3) {
    switch(op) {
      case 0: mode=a; break;
      case 1: mode=a; width=b; break;
      case 2: if (mode!=a) {status=17;goto done;} obs+=b; break;
      case 3: mode=-31; break;
      case 4: mode=a;pulses++;break;
      case 5: mode=-41;obs+=19;break;
      case 6: obs+=a;break;
      default: return 2;
    }
  }
done:
  printf("%lld %lld %lld %lld %lld\n",status,mode,width,obs,pulses);
  return 0;
}
"""


@pytest.mark.parametrize(
    "case",
    [
        "basic",
        "multiple-fields",
        "partial-redundancy",
        "clobber",
        "unknown-barrier",
        "unordered-barrier",
        "retained-observers",
        "unknown-read",
    ],
)
def test_native_explicit_fixture_complete_outcomes(case, tmp_path):
    _, _, _, projection = produce(tmp_path, CASES[case])
    compiler = shutil.which("cc")
    if compiler is None:
        state._write(tmp_path / "native.json", {"status": "unavailable", "reason": "cc absent"})
        pytest.skip("explicit native fixture compiler unavailable")
    source, binary = tmp_path / "fixture.c", tmp_path / "fixture"
    source.write_text(NATIVE)
    build = subprocess.run(
        [compiler, "-std=c11", "-O2", "-Wall", "-Werror", str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    (tmp_path / "native-build.stdout").write_text(build.stdout)
    (tmp_path / "native-build.stderr").write_text(build.stderr)
    assert build.returncode == 0, build.stderr

    def commands(atoms):
        rows = []
        for atom in atoms:
            name, args = operands(atom)
            if name in ("setup", "pulse", "observe"):
                row = ({"setup": 0, "pulse": 4, "observe": 6}[name], args["value"], 0)
            elif name == "pair":
                row = (1, args["mode"], args["width"])
            elif name == "work":
                row = (2, args["required"], args["add"])
            else:
                row = (3 if name == "reset" else 5, 0, 0)
            rows.append(" ".join(map(str, row)))
        return "\n".join(rows) + "\n"

    outcomes = []
    for label in ("before", "after"):
        data = commands(projection[label])
        (tmp_path / (label + ".commands")).write_text(data)
        run = subprocess.run([str(binary)], input=data, text=True, capture_output=True, timeout=10)
        (tmp_path / (label + ".native.stdout")).write_text(run.stdout)
        (tmp_path / (label + ".native.stderr")).write_text(run.stderr)
        assert run.returncode == 0
        outcomes.append(run.stdout)
    state._write(
        tmp_path / "native.json",
        {
            "status": "fixture_compared",
            "compiler": state.file_ref(compiler),
            "binary": state.file_ref(binary),
            "source": state.file_ref(source),
            "outcomes": outcomes,
            "actual_projection": state.file_ref(tmp_path / "actual/projection.json"),
            "hardware_fidelity_proved": False,
            "Lean_native_bridge_proved": False,
        },
    )
    assert outcomes[0] == outcomes[1]
