# Security finding #1116: Compiler-controlled ONNX is passed to CreateSessionFromArray withou…

**Full report, discussion and status: https://github.com/ibelem/chromium/issues/81**

| | |
| --- | --- |
| CWE | CWE-829: Inclusion of Functionality from Untrusted Control Sphere |
| Location | [`targets/chromium/services/webnn/ort/graph_impl_ort.cc:346`](https://github.com/chromium/chromium/blob/cdbdcb205e3483f419c7d5a7dc56a048cddc228c/services/webnn/ort/graph_impl_ort.cc#L346-L348) — `GraphImplOrt::CreateSessionFromCompiledGraph()` |
| Trust boundary | Compromised WebNN Compiler process to the GPU process over WebNNModelLoader.LoadCompiledGraph. |
| Validated for | chromiumWebnn |
| Reachable from | chromiumWebnn |

This file is an index entry only — it deliberately does not duplicate the
issue. See https://github.com/ibelem/chromium/issues/81 for the root cause, exploit, reproduction and
suggested fix.


Models: ideaCreator `openai,gpt-5.6-cyber`, techLead `openai,gpt-5.6-cyber`, researcher `openai,gpt-5.6-cyber`, validator `openai,gpt-5.6-cyber`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #1116.