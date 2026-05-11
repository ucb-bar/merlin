---
section: acceptance
merge: replace
---
{% for check in task_acceptance_checks %}
- {{ check }}
{% endfor %}

Validation commands:
{% for command in task.validation_commands %}
- `{{ command }}`
{% endfor %}
