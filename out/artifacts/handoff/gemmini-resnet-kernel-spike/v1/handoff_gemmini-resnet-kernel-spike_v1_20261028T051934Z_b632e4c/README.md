# ResNet and kernel Spike verification delivery

Source/build entry points: Merlin `b632e4c97bcc5e1d05881db41aa5e22360740601`, TVM `bb4ad9f8dbd21dd96a4dc63121dfebc24742f260`.
The **74 operator fixtures, 16 complete kernels, 18 ResNet chunks and connected
ResNet parent** pass diagnostic Spike. Every ELF pair also rejects the deliberately
wrong reference. See `summary.json` for hashes and scope, and `evidence.tar.gz`
for original receipts, full placement records, logs, reference identities and tests.

```sh
sha256sum -c SHA256SUMS
mkdir evidence
 tar -xzf evidence.tar.gz -C evidence
```

The archive contains evidence, **not prebuilt ELFs or frozen input tensors**.
Build them using the tracked instructions at
`examples/gemmini/comparisons/tvm/FIRESIM_HANDOFF.md`. The kernel generators and
references are committed Python source. ResNet uses the existing committed,
checksum-verified split archive through `restore_resnet_inputs.py`. No author's
private `out/` directory, Git-ignored ELF plan or unpublished model bundle is needed.
Original absolute paths in receipts/extraction records are historical identities;
regenerate the study using `prepare_active_study.py` on the remote, rather than
executing those original-path study files.

The evidence archive preserves emitted Spike counters only as raw diagnostic
logs; **none is a FireSim performance measurement**. The oracle is the repaired
RNE diagnostic provider, not the missing provider pinned by the software spec.
The full ResNet check validates the published parent graph against its independent
reference; its frozen numerical proof still requires exact selected-protocol
admission before a selected-model claim. Deployed hardware and cancellation
qualification remain remote work.

The first C01 form fell back to CPU and is explicitly excluded. The admitted
flat-output TensorIR form keeps exactly the same inputs/references and executes
on Gemmini. All 16 admitted cases reject CPU fallback. The measured profile to
use remotely is `warm_repeat`; historical 2 MiB cache-eviction results are distinct.

Focused tests: **36 passed**. Repository checks: no-target-name passes; structure
retains nine pre-existing module-size failures; the full no-regex scan reports
eight existing call sites in unchanged core/research files. Staged no-regex passes.
Full TinyLlama and SmolVLA are excluded from this campaign.
