# Spectformer r5 OPU source read set

The delivered r5 ZIP preserves the exact eight model ELFs and VLEN probe from r4. Its
`manifest.json` records Saturn commit `e16a301d261fa4b5a374eb7069b78107a8c9aefb` and
`unit_source_digest=da81632bbf22d373a291f862dce0c1ab2098338417b3cc892fb2fb32e9aee03d`.
The declared Saturn pin is an older revision, so the manifest marks the unit revision unverified.

On 2026-11-06, each file below matched `git show e16a301d:<path>` byte for byte:

| Saturn relative path | SHA-256 |
| --- | --- |
| `benchmarks/common/bme.h` | `5aad9df38a40cdee7645183c13ad0ed97cc32f3da677a6c441f073fe87e278f4` |
| `src/main/scala/common/Consts.scala` | `aa53425eec1492d5ac2514c9c132ff96df1d2e9f6ce897fc48dba95d9060a1fe` |
| `src/main/scala/common/Parameters.scala` | `70e5881efdcf17473902fc57119fbfdb61e1e41ad2123e32585ed7386ad8a2e5` |
| `src/main/scala/insns/Instructions.scala` | `b8b2c6e9c2a598cc64025eb58fcf85fb850622915ddb3c1fa783c323f24cc083` |

Those committed bytes reproduce the manifest digest when paired with its build host paths.
That aggregate digest includes absolute paths, so compare the individual hashes on another
machine. The manifest's `dirty_files: 7` counts repository status entries; at this audit they
were only untracked benchmark logs and cache, with no tracked Saturn source changes.

This read set establishes the OPU instruction and geometry source used by the package. It does
not prove a byte-identical ELF rebuild: the archive lacks build-time Chipyard, Zephyr, and
toolchain pins, and the capture receipt verifies artifacts but not the model source closure.
Use the delivered ELFs for exact bytes. Gate any new build independently on Spike and hardware.
