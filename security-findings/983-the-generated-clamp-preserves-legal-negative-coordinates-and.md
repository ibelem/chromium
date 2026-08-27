# Security finding #983: The generated clamp preserves legal negative coordinates and passes…

**Full report, discussion and status: https://github.com/ibelem/chromium/issues/66**

| | |
| --- | --- |
| CWE | CWE-20: Improper Input Validation |
| Location | [`targets/chromium/services/webnn/ort/graph_builder_ort.cc:2931`](https://github.com/chromium/chromium/blob/95bf246499c3d2bcc5d20e029d404fc4b67ba507/services/webnn/ort/graph_builder_ort.cc#L2931-L2940) — `GraphBuilderOrt::AddScatterNDOperation()` |
| Trust boundary | Compromised renderer-controlled ScatterND tensor through Mojo, ORT and OpenVINO EP to Intel NPU |
| Validated for | chromiumWebnn |
| Reachable from | chromiumWebnn |

This file is an index entry only — it deliberately does not duplicate the
issue. See https://github.com/ibelem/chromium/issues/66 for the root cause, exploit, reproduction and
suggested fix.


Models: ideaCreator `openai,gpt-5.6-cyber`, techLead `openai,gpt-5.6-cyber`, researcher `openai,gpt-5.6-cyber`, validator `openai,gpt-5.6-cyber`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #983.