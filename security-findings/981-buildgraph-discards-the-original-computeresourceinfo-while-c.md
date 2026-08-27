# Security finding #981: BuildGraph discards the original ComputeResourceInfo, while Compile…

**Full report, discussion and status: https://github.com/ibelem/chromium/issues/63**

| | |
| --- | --- |
| CWE | CWE-345: Insufficient Verification of Data Authenticity |
| Location | [`targets/chromium/services/webnn/ort/compiler_context_impl_ort.cc:78`](https://github.com/chromium/chromium/blob/95bf246499c3d2bcc5d20e029d404fc4b67ba507/services/webnn/ort/compiler_context_impl_ort.cc#L78-L82) — `CompilerContextImplOrt::BuildGraph()` |
| Trust boundary | Sandboxed WebNN Compiler process to GPU process |
| Validated for | chromiumWebnn |
| Reachable from | chromiumWebnn |

This file is an index entry only — it deliberately does not duplicate the
issue. See https://github.com/ibelem/chromium/issues/63 for the root cause, exploit, reproduction and
suggested fix.


Models: ideaCreator `openai,gpt-5.6-cyber`, techLead `openai,gpt-5.6-cyber`, researcher `openai,gpt-5.6-cyber`, validator `openai,gpt-5.6-cyber`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #981.