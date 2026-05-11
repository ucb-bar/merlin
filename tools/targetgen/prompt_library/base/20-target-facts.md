---
section: target_facts
merge: replace
---
- Display name: `{{ target.display_name }}`
- Vendor: `{{ target.vendor }}`
- Maturity: `{{ target.maturity }}`
- Host ISA: `{{ platform.host_isa }}`
- Execution model: `{{ execution.kind }}`
- ISA exposure: `{{ isa.exposure.kind }}`
- Access model: `{{ capabilities.access.model }}`
- SDK requirements: `{{ capabilities.access.sdk_requirements | comma_list }}`
- Credential requirements: `{{ capabilities.access.credential_requirements }}`
- Availability class: `{{ capabilities.access.availability_class }}`
- Verification gates: `{{ capabilities.access.verification_gates | comma_list }}`
- Target families: `{{ support_plan.target_families | comma_list }}`
- Integration styles: `{{ support_plan.integration_styles | comma_list }}`
- Primary integration: `{{ support_plan.primary_integration }}`
- Required layers: `{{ support_plan.required_layers | comma_list }}`
{% if deployment %}
- Deployment profile: `{{ deployment.name }}` (`{{ deployment.mode }}`, build profile `{{ deployment.build_profile }}`)
{% endif %}
