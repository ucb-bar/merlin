# Required-inputs manifest (all)

> The inherent limits that remain, each with the EXACT input/run that closes it. These are scoped (not bare caveats) and nothing is fabricated — Merlin reports what a real DSE run still needs.

| limit | evidence today | required input to close |
|---|---|---|
| real deployment K + control rate | Tier C | a deployment/runtime trace giving actual loop counts + control frequency |
| per-unit throughput / latency / area / energy | unavailable | a candidate design YAML (unit shapes + a cost model); then the future DSE tool computes them |
| KV / attention structure + true data deps at loop level | unavailable | a Level-2 loop-preserving, attention-not-lowered capture |
| packed low-bit layout + scales for the recaptured models | low-bit storage shown on the zoo (numerical_contract_fidelity_report.md) | a low-bit (packed weights + scale metadata) capture of the recaptured models |
| fp8 / int4 accuracy gates | int8 measured | per-format accuracy runs (W8A8 already done) for fp8 / int4 |
| real-magnitude weights | structure recovered_from_ir | full-size (non-random-init) captures of the same architectures |
