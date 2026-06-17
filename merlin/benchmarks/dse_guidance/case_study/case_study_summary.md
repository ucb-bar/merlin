# Case study summary — workload-contract analysis

> Flat captures are not DSE-ready workload descriptions. Merlin recovers a temporal/numerical workload contract from provenance-rich captures and emits HW/SW abstraction candidates + hardware-independent requirements for a future DSE engine — no speedup claimed.

| workload | recovered roles | real IR facts | derived requirement | implied abstraction | missing before DSE |
|----------|-----------------|---------------|---------------------|---------------------|--------------------|
| rdt | repeated_head:20 | 197 GMAC/replan, 1.56 GB avoidable reload | resident bf16 196 MB | resident_weight_object | quantization accuracy gates (per candidate low-bit format); real (target) command-submit / sync latency; K / control-rate from the real deployment (currently reference values) |
| openvla | backbone_once:11, repeated_head:15 | 0 GMAC/replan, 0.02 GB avoidable reload | resident bf16 2 MB | resident_weight_object | real (target) command-submit / sync latency; K / control-rate from the real deployment (currently reference values) |
| small_llama | repeated_head:15 | 0 GMAC/replan, 0.05 GB avoidable reload | resident bf16 1 MB | resident_weight_object | real (target) command-submit / sync latency; K / control-rate from the real deployment (currently reference values) |
| tiny_llama | repeated_head:15 | 20 GMAC/replan, 19.05 GB avoidable reload | resident bf16 307 MB | resident_weight_object | real (target) command-submit / sync latency; K / control-rate from the real deployment (currently reference values) |

See per-workload `<workload>/workload_contract_report.md` for the full package, `requirements_table.csv` / `dtype_capacity_table.csv` for requirements, `abstraction_pressure_table.csv` for the HW/SW abstractions, `dse_readiness_summary.csv` for readiness, and `accuracy_gate_report.md` for the measured int8 accuracy leg.
