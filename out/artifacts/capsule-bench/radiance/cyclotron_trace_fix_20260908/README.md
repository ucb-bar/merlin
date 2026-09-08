# Cyclotron trace-disabled memory fix replay

This receipt records a controlled before/after replay of the same compiled
Radiance capsule. Both arms ran the same ELF and generated Cyclotron config;
the only changed executable was Cyclotron itself.

- Capsule: `SY_app_contraction_f32_partial_unknown_rank3_odd_tail_heavy_l2`
- ELF SHA-256: `cb3ba94d175381876bd6c02909df26b30c8e643c512293bc9f38fd256ba6d3d0`
- Config SHA-256: `680ef78b59bd728620465ca5e0d357f30dfb53b267fab9ca202b71691c6a3212`
- Source parent: `3fd83f2bb608495ca6ffbed38043cdbadb215722`
- Fix commit: `2d6adad4ad94d1621fdff9c9a1e5eac871048f24`

The source parent used 624,000 KiB maximum RSS. The fixed binary used 6,912
KiB: a 90.28x reduction. Both runs printed exactly
`MERLIN_NUMERIC PASS 8150 0`, `DONE`, and `simulation finished after 1112993
cycles`. Wall time was effectively unchanged (5.98 s before, 6.05 s after).

The original failing N=1024 observation used the previously selected binary
SHA-256 `bca96b105be847577693bf6e16a0d7ef82dda2d41703aa9edffe19aee6a77326`
and reached exactly the configured 20,000,000-cycle cap with 20,009,440 KiB
maximum RSS. That binary was overwritten by the rebuild, so the controlled
before arm was rebuilt from the fix commit's immediate source parent and is
identified separately in `receipt.json`.

The replay command template was:

```sh
/usr/bin/time -v <cyclotron-binary> <generated>/cyclotron.run.toml \
  --binary-path <generated>/kernel.radiance.elf --timing --log 0
```

Raw replay console/time files remain beside the generated ELF under
`focused_n163_allwarps_v2/.../generated/trace_fix_proof/`; their hashes are
sealed in the receipt so generated-output cleanup is detectable.
