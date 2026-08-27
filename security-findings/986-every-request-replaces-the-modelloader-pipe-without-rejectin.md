# Security finding #986: Every request replaces the ModelLoader pipe without rejecting an ex…

**Full report, discussion and status: https://github.com/ibelem/chromium/issues/72**

| | |
| --- | --- |
| CWE | CWE-400: Uncontrolled Resource Consumption |
| Location | [`targets/chromium/services/webnn/ort/dispatch_context_impl_ort.cc:108`](https://github.com/chromium/chromium/blob/95bf246499c3d2bcc5d20e029d404fc4b67ba507/services/webnn/ort/dispatch_context_impl_ort.cc#L108-L109) — `DispatchContextImplOrt::RequestCompilerContext()` |
| Trust boundary | Compromised renderer Mojo WebNNContext to the shared per-device compiler utility process |
| Validated for | chromiumWebnn |
| Reachable from | chromiumWebnn |

This file is an index entry only — it deliberately does not duplicate the
issue. See https://github.com/ibelem/chromium/issues/72 for the root cause, exploit, reproduction and
suggested fix.


Models: ideaCreator `openai,gpt-5.6-cyber`, techLead `openai,gpt-5.6-cyber`, researcher `openai,gpt-5.6-cyber`, validator `openai,gpt-5.6-cyber`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #986.