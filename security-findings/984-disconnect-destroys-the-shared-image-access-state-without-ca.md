# Security finding #984: Disconnect destroys the shared-image access state without cancellin…

**Full report, discussion and status: https://github.com/ibelem/chromium/issues/69**

| | |
| --- | --- |
| CWE | CWE-362: Concurrent Execution using Shared Resource with Improper Synchronization |
| Location | [`targets/chromium/services/webnn/webnn_tensor_impl.cc:260`](https://github.com/chromium/chromium/blob/95bf246499c3d2bcc5d20e029d404fc4b67ba507/services/webnn/webnn_tensor_impl.cc#L260-L263) — `WebNNTensorImpl::OnDisconnect()` |
| Trust boundary | Renderer-controlled WebNNTensor Mojo pipe teardown versus WebNN GPU scheduler |
| Validated for | chromiumWebnn |
| Reachable from | chromiumWebnn |

This file is an index entry only — it deliberately does not duplicate the
issue. See https://github.com/ibelem/chromium/issues/69 for the root cause, exploit, reproduction and
suggested fix.


Models: ideaCreator `openai,gpt-5.6-cyber`, techLead `openai,gpt-5.6-cyber`, researcher `openai,gpt-5.6-cyber`, validator `openai,gpt-5.6-cyber`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #984.