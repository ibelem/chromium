# Security finding #979: OnLost has no already-lost guard, so every queued failed tensor rea…

**Full report, discussion and status: https://github.com/ibelem/chromium/issues/58**

| | |
| --- | --- |
| CWE | CWE-617: Reachable Assertion |
| Location | [`targets/chromium/services/webnn/webnn_context_impl.cc:636`](https://github.com/chromium/chromium/blob/95bf246499c3d2bcc5d20e029d404fc4b67ba507/services/webnn/webnn_context_impl.cc#L636-L646) — `WebNNContextImpl::OnLost()` |
| Trust boundary | Compromised renderer to WebNN service Mojo/data-pipe boundary |
| Validated for | chromiumWebnn |
| Reachable from | chromiumWebnn |

This file is an index entry only — it deliberately does not duplicate the
issue. See https://github.com/ibelem/chromium/issues/58 for the root cause, exploit, reproduction and
suggested fix.


Models: ideaCreator `openai,gpt-5.6-cyber`, techLead `openai,gpt-5.6-cyber`, researcher `openai,gpt-5.6-cyber`, validator `openai,gpt-5.6-cyber`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #979.