---
section: evidence
merge: replace
---
{% for item in task_evidence %}
- `{{ item.value }}`: {{ item.reason }}
{% endfor %}
