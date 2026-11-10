# Muon host support for Merlin

This local-only OOT repository owns the Muon backend, its native carrier resource,
and Cyclotron oracle integration previously stored in Merlin. No publication
remote is configured, and no release or compiler qualification is implied.

Select this repository root explicitly:

```sh
export MERLIN_TARGET_PATH=/absolute/path/to/muon-support
```

The provider and backend identity is `muon`. The Radiance experiment retains its
own `radiance` identity; these are not interchangeable target aliases. The copied
contract describes support, not a generated compiler candidate. The source
comments mentioning implicit discovery are historical: current Merlin requires
explicit provider selection.

The backend reuses Merlin's shared IR, numerical, toolchain and fixed-format
machinery. Code generation and oracle methods may launch expensive native
compilers or simulators. Import and pure-source tests do not authorize those runs.
Keep the whole support root host-private during compiler evaluation.

The `provenance.json` inventory commits the original 19 files, including
`backend/soc_carrier/main.c`, to the exact Merlin source revision and SHA-256.
No runtime behavior was changed by copying those bytes. A preceding source fix
makes build-cache identity use the installed lowering owner rather than a legacy
checkout path; it does not claim complete transitive toolchain attribution.

No RTL facts or IRDL pins are bundled. Configure reviewed facts explicitly;
missing facts must not borrow same-name native evidence. Radiance's retained
descriptor still declares its historical reference metadata/pin location. Moving
support does not populate that location, qualify a fresh bundle, or authorize
resuming an old frozen run. Resource/pin declaration migration and independent
native execution remain unfinished.
