"""Adjacent OpenMP region coarsening preserves worksharing barriers and block boundaries."""

from merlin.llvmlower import impr_features
from merlin.llvmlower.accum_microkernel import run_source as accumulator_runner_source
from merlin.llvmlower.parallel_coarsen import (
    FEATURE,
    apply_for_test,
    ensure_registered,
    require_report,
)
from merlin.llvmlower.pipeline import _RUNNER_SRC, _activation_poly_runner


def test_feature_is_named_and_lazy_resolvable():
    assert ensure_registered() == FEATURE
    assert FEATURE in impr_features.normalize([FEATURE])


def test_every_feature_specific_runner_carries_the_coarsening_stage():
    for source in (_RUNNER_SRC, _activation_poly_runner(), accumulator_runner_source()):
        assert "_PARALLEL_COARSEN" in source
        assert "_parallel_coarsen" in source


def test_adjacent_regions_merge_but_intervening_work_splits_runs():
    module, report = apply_for_test(r'''module {
      func.func private @side_effect()
      func.func @forward(%buf: memref<1xi64>) {
        %c0 = arith.constant 0 : i64
        %c1 = arith.constant 1 : i64
        omp.parallel {
          omp.wsloop {
            omp.loop_nest (%i) : i64 = (%c0) to (%c1) step (%c1) {
              %idx = arith.index_cast %i : i64 to index
              memref.store %c1, %buf[%idx] : memref<1xi64>
              omp.yield
            }
          }
          omp.terminator
        }
        omp.parallel {
          omp.wsloop {
            omp.loop_nest (%i) : i64 = (%c0) to (%c1) step (%c1) {
              %idx = arith.index_cast %i : i64 to index
              memref.store %c0, %buf[%idx] : memref<1xi64>
              omp.yield
            }
          }
          omp.terminator
        }
        func.call @side_effect() : () -> ()
        omp.parallel {
          omp.wsloop {
            omp.loop_nest (%i) : i64 = (%c0) to (%c1) step (%c1) {
              %idx = arith.index_cast %i : i64 to index
              memref.store %c1, %buf[%idx] : memref<1xi64>
              omp.yield
            }
          }
          omp.terminator
        }
        return
      }
    }''')
    assert report == {"merged": 1}
    assert module.count("omp.parallel") == 2
    assert module.count("omp.wsloop") == 3
    assert module.index("omp.wsloop") < module.index("call @side_effect")


def test_noncanonical_parallel_operands_fail_closed():
    module, report = apply_for_test(r'''module {
      func.func private @side_effect()
      func.func @forward() {
        %c2 = arith.constant 2 : i32
        omp.parallel num_threads(%c2 : i32) {
          func.call @side_effect() : () -> ()
          omp.terminator
        }
        omp.parallel num_threads(%c2 : i32) {
          func.call @side_effect() : () -> ()
          omp.terminator
        }
        return
      }
    }''')
    assert report == {"merged": 0}
    assert module.count("omp.parallel") == 2


def test_named_build_requires_a_nonempty_runner_report():
    require_report("OK parallel_coarsen original 3 merged 2 groups 1 remaining 1")
    import pytest
    with pytest.raises(ValueError, match="did not execute"):
        require_report("")
    with pytest.raises(ValueError, match="matched no adjacent"):
        require_report("OK parallel_coarsen original 3 merged 0 groups 0 remaining 3")
