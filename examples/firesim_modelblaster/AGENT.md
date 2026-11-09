# FireSim ModelBlaster runner example

This independently installable package is an out-of-tree FireSim execution
adapter retained for external operators and research reproduction. It is not
part of the Merlin core wheel. Core prepares/collects and no longer loads the
legacy runner entry point; operator invocation remains outside that API.

Keep ModelBlaster checkout discovery, workload/project identity, queue policy,
and simulator imports here. Do not put executable support in
`examples/saturn/target/`, which is reference metadata only. Never claim an
FPGA result from the adapter's import or preflight alone.
