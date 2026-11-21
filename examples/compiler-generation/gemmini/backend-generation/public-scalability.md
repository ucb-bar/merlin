# Required public Spike scalability suite

This is an opt-in selection for a **new** release. It leaves the core public RTL
cohort, its score and direct certification floor unchanged. Hidden validation
and full-model validation remain separate after the compiler is frozen.

Select a reviewed native capsule directory with `corpus prepare
--public-scalability /absolute/path/suite.json` (or the Python API's
`public_scalability=Path(...)`). The JSON selector has these fields:

- `schema`: `merlin.corpus.public_scalability.v1`.
- `target`: the selected descriptor's target; `origin`: `operator_requested`.
- `policy`: exactly `{"required":true,"engine":"spike","tiers":["L0","L1","L2"]}`.
- `provenance`: an absolute ordinary file `path` and its `sha256`.
- `members`: the complete list of `{member,sha256,source_group,split}` records.
  `member` is `category/name`; its digest covers the entire capsule directory,
  including sources, inputs, references and numerical policy. `source_group` is
  the predetermined source-family SHA256 identity and `split` is `public` or
  `train`. Variants in one source group must retain the same split. Identical
  mathematical sources cannot be relabeled as different groups.

Every member must already have its native public mathematical/numerical contract
and accelerator placement. It must require exactly L0/L1/L2, without a deeper-tier
ceiling, substitute sibling, performance policy or expected refusal. Pending
attention/state/effect contracts are not admitted. Selecting this input does not
claim it came from automatic Phase0 generation or establish execution success.
The existing public/held-out source and private-alias guards remain mandatory.

Preparation freezes the complete suite privately. Only mathematical IR and an
answer-free source/policy roster enter the author grants. Goldens, arrays and
producer metadata stay private. The author receives ordinary helper instructions:

```sh
python agent_selfcheck.py --suite scalability --capsules NAME1,NAME2 --wait-budget 5
python agent_selfcheck.py --attach REQUEST_ID
python agent_selfcheck.py --suite scalability --capsules all --wait-budget 5
```

Named subsets use existing asynchronous broker/concurrency support. Their actual
case outcomes are useful feedback; they cannot complete the full denominator.
The broker and worker reject gSIM, L3, alternate engines and inspection modes for
this suite. They use the ordinary Spike grader, complete-output comparison,
placement and source/instruction checks under the original compiler execution
profile. Existing per-program compilation/device-IR bounds and runtime resource
limits remain binding; unavailable or timed-out cases remain incomplete.

A complete core convergence candidate and final publication must also pass every
selected suite case on the same candidate and pinned inputs. This first version
adds no cross-run runtime cache: a later convergence/publication can repeat the
full suite. Exact counts and the distinct observed producer are retained; the
suite supplies no RTL or timing credit. It cannot replace hidden/full-model gates
or add anything to the Phase2 scored/cohort/timing records.

No complete large-suite selection is shipped here. The public Phase2 source
inventory (including the named benchmark subset) still needs its exact
source/split/custody and numerical/state admission joins before it can become
this selector. Neither an inventory count nor historical timing is admission.
The two prior one-time full-shape L3 examples remain separate explicit selections.
