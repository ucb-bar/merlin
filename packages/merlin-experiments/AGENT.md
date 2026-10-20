# Merlin experiments package

Owns versioned experiment definitions, typed phase adapters, and orchestration.
Keep grades, numerical semantics, native resume gates, and AET accounting in
their existing engines. Importing this package or inspecting a definition must
never launch an agent, simulator, or hardware job. Public definitions cannot
contain arbitrary shell commands.

Run `python -m pytest packages/merlin-experiments/tests` from the checkout.
Synthetic process tests must not require credentials, paid agents, or hardware.

The native instruction and numerical audit modules share the original exact-ELF
ISA and all-output rechecks across explicit execution ABIs. They grant no source
placement, execution-effect, RTL, timing or promotion authority.
