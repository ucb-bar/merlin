# Build and repository tools

This directory owns repository checks, package construction, documentation
renderers and deployment tools used by current workflows. Start at
`README.md`. Compiler and evaluation implementations belong in `src/merlin/`.

Keep generated files, installed toolchains, run data and private review notes
under ignored `out/` or `tmp/`. Do not add dated campaigns, publication figures,
captured results, empty scaffolds or another general-purpose scripts folder.

Preserve reviewed policy and ledger entries when moving tools. A new path cannot
clear existing debt or reset its baseline. Shared check helpers have one owner in
`checks/`; generators import that owner rather than copying it.

Packaging checks must inspect the actual wheel/source archive and installed
imports outside the checkout. Execute every selected qualification member and
retain refusals and skips. Software checks do not establish hardware execution.
