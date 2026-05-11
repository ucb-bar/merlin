---
section: response_contract
merge: replace
---
Respond with:

- what you changed or validated
- any unresolved blockers or assumptions
- the exact verification you ran, or why you could not run it
- the file paths you edited when code changed
- the produced artifacts in `{{ task.artifacts_out | comma_list }}`
- whether credentials beyond `{{ task.credential_requirements }}` were needed

Handoff requirements:
{% for item in task.handoff_contract %}
- {{ item }}
{% endfor %}
