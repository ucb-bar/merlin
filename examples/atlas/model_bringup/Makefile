# The execution view is outside the repositories and always verifies canonical bytes.
PYTHON ?= python3
MERLIN_ROOT ?= $(abspath ../../..)
VIEW ?= $(abspath $(MERLIN_ROOT)/../out/model-bringup-view)
RESULTS_ROOT ?= $(abspath $(dir $(VIEW))/model-bringup-results)
MODE ?= relocated

.PHONY: check view test faithfulness
check:
	@mkdir -p "$(RESULTS_ROOT)"; a=0; b=0; $(MAKE) test || a=$$?; $(MAKE) faithfulness || b=$$?; \
	  printf '{"test_exit":%s,"faithfulness_exit":%s}\n' "$$a" "$$b" > "$(RESULTS_ROOT)/check-status.json"; \
	  test $$a -eq 0 -a $$b -eq 0

view:
	@test -n "$(ATLAS_ROOT)" -a -n "$(KERNELS_ROOT)" || { echo 'Set ATLAS_ROOT and KERNELS_ROOT to explicit canonical roots' >&2; exit 1; }
	@if test -f "$(VIEW).inventory.json"; then \
	  PYTHONDONTWRITEBYTECODE=1 "$(PYTHON)" source_view.py --verify --view "$(VIEW)" --atlas "$(ATLAS_ROOT)" --merlin "$(MERLIN_ROOT)" --kernels "$(KERNELS_ROOT)" --mode "$(MODE)"; \
	else \
	  PYTHONDONTWRITEBYTECODE=1 "$(PYTHON)" source_view.py --atlas "$(ATLAS_ROOT)" --merlin "$(MERLIN_ROOT)" --kernels "$(KERNELS_ROOT)" --view "$(VIEW)" --mode "$(MODE)"; \
	fi

# Both targets fail on skipped required checks and retain separate receipts.
test: view
	PYTHONDONTWRITEBYTECODE=1 "$(PYTHON)" run_tests.py tests $(if $(TINY_CODEC_MODEL_ROOT),--codec-model-root "$(TINY_CODEC_MODEL_ROOT)") --mode "$(MODE)" --view "$(VIEW)" --atlas "$(ATLAS_ROOT)" --merlin "$(MERLIN_ROOT)" --kernels "$(KERNELS_ROOT)" $(if $(MODEL_ROOT),--model-root "$(MODEL_ROOT)") --out "$(RESULTS_ROOT)/tests.json"

faithfulness: view
	PYTHONDONTWRITEBYTECODE=1 "$(PYTHON)" run_tests.py faithfulness --mode "$(MODE)" --view "$(VIEW)" --atlas "$(ATLAS_ROOT)" --merlin "$(MERLIN_ROOT)" --kernels "$(KERNELS_ROOT)" $(if $(MODEL_ROOT),--model-root "$(MODEL_ROOT)") --out "$(RESULTS_ROOT)/faithfulness.json"
