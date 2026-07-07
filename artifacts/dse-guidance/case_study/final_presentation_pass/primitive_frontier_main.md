# Primitive-set frontier (`primitive_set_frontier`)

Source: `IM.primitive_set_frontier` over `primitive_coverage_matrix.csv`.

- **Coverage** = the fraction of a workload's MACs that a primitive (or set) can tile at **≤ 10% pad waste**.
  A set covers an op if **any** member tiles it under the threshold. This is **structural pad-waste coverage
  only — it does not rank hardware performance.**
- Axes: x = mean coverage across workloads; y = worst-workload coverage. Upper-right = broadly useful.
- The plot annotates the **best size-1, size-2, size-3** sets (markers ●/★/✚) and the single primitives (dots).
- **Takeaway:** the best single primitive leaves the **worst** workload poorly covered; moving to a **2-set**
  jumps worst-case coverage to ~full, and **3 adds little** — so a small primitive *set* is the DSE axis, not
  a single primitive.
- Title: *"Primitive-set frontier: one primitive is not robust across workloads."* Caveat:
  *"structural pad-waste MAC coverage — does not rank hardware performance."*
