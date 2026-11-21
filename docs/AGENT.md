# Documentation

The public handbook has twelve numbered chapters following compilation and
execution, then compiler generation and experiments. Start at `README.md`.
Keep one explanation per concept and link to its source owner.

Update an existing chapter before adding another page. Use simple technical
language and real commands. Distinguish implemented behavior from design ideas.
Do not put dated audits, trial reports or runtime data in this tree.

Chapters and handwritten references use YAML front matter with title, kind,
status, owner, last_verified, related and actual code_refs. Update the authored reading index and run `build_tools/checks/check_docs.py`.
Generated CLI, module and schema references belong under ignored `out/`, not here.
