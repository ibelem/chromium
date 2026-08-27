# Security finding #980: A renderer-provided release_count is converted into a service-owned…

**Full report, discussion and status: https://github.com/ibelem/chromium/issues/61**

| | |
| --- | --- |
| CWE | CWE-667: Improper Locking |
| Location | [`targets/chromium/services/webnn/webnn_tensor_impl.cc:185`](https://github.com/chromium/chromium/blob/95bf246499c3d2bcc5d20e029d404fc4b67ba507/services/webnn/webnn_tensor_impl.cc#L185-L190) — `WebNNTensorImpl::ExportTensor()` |
| Trust boundary | Compromised renderer-controlled release_count crosses Mojo into the WebNN GPU service scheduler. |
| Validated for | chromiumWebnn |
| Reachable from | chromiumWebnn |

This file is an index entry only — it deliberately does not duplicate the
issue. See https://github.com/ibelem/chromium/issues/61 for the root cause, exploit, reproduction and
suggested fix.


Models: ideaCreator `openai,gpt-5.6-cyber`, techLead `openai,gpt-5.6-cyber`, researcher `openai,gpt-5.6-cyber`, validator `openai,gpt-5.6-cyber`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #980.