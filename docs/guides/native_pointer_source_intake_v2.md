# Prospective bounded native source storage

The original owner may explicitly select
`source_intake_protocol="merlin.native_pointer_source_intake.v2"` together with
the prospective packet V3 source checker. The generated-source factory pins this
selection and its implementation. The checked-source consumer carries the live
selection through execution and both numerical/source audits. Select the same
protocol for the control and candidate. Neither a command-buffer field nor a
saved result selects it.

This changes software storage transport only. The logical pointer ABI remains
`merlin.native_pointer_source.v1`; source arithmetic, complete K, final signed
saturation, all-output equality, hardware, compiler and timing contracts retain
their independent checks. Native i8 results require their original typed i8 exact
policy and packet V3 saturation proof; raw i32 controls keep their exact i32 policy.

The selected bounds are 16,000,000 elements per tensor, 256 MiB of complete
logical storage and decoded payload per archive/namespace, and a conservative
512 MiB working estimate. Declarations, dense storage geometry and NPY headers
are checked before array allocation. The estimate includes every reference and
captured input byte plus 64 bytes per output element for existing complete Python
readback decoding. It leaves half of the campaign's unchanged 1 GiB worker limit
for Python, source owners and builds. A shape below the element ceiling can still
be refused by the byte or working bounds.

The six public GEMMs include 3,211,264-element operands/results; the first public
convolution's explicit im2col has 7,225,344 elements, and existing width-2048
controls have 4,194,304-element square tensors. These extents motivate the new
ceiling without removing a finite bound. Input transport uses immutable typed
byte payloads, references use digest-checked read-only NumPy arrays, and exact
comparison visits every output scalar in chunks of at most 65,536 cells. No
per-MAC proof rows or sampled outputs are introduced.

The same explicit selection lets the neutral native harness use at most 256 MiB
of logical storage. It does not edit the frozen contract, and any smaller
contract-declared storage budget still applies. Without this selection, the
original one-million-element pointer intake and 16 MiB payload/harness defaults
and refusals remain in force.
