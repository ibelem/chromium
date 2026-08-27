# Security finding #978: Outer ORT descriptors are accepted without comparison to embedded O…

**Full report, discussion and status: https://github.com/ibelem/chromium/issues/56**

| | |
| --- | --- |
| CWE | CWE-20: Improper Input Validation |
| Location | [`targets/chromium/services/webnn/ort/graph_impl_ort.cc:354`](https://github.com/chromium/chromium/blob/95bf246499c3d2bcc5d20e029d404fc4b67ba507/services/webnn/ort/graph_impl_ort.cc#L354-L361) — `GraphImplOrt::CreateSessionFromCompiledGraph()` |
| Trust boundary | Sandboxed Compiler process to GPU process via WebNNModelLoader |
| Validated for | chromiumWebnn |
| Reachable from | chromiumWebnn |

This file is an index entry only — it deliberately does not duplicate the
issue. See https://github.com/ibelem/chromium/issues/56 for the root cause, exploit, reproduction and
suggested fix.


Models: ideaCreator `openai,gpt-5.6-cyber`, techLead `openai,gpt-5.6-cyber`, researcher `openai,gpt-5.6-cyber`, validator `openai,gpt-5.6-cyber`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #978.