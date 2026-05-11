---
section: implementation_focus
merge: replace
---
{% for action in task_actions %}
- {{ action }}
{% endfor %}

- Primary repo root: `{{ task.repo_root }}`
- Execution adapter: `{{ task.execution_adapter }}`
- Mutation policy: `{{ task.mutation_policy }}`
