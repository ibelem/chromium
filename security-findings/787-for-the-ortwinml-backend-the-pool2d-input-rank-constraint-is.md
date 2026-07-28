# Security finding #787: For the ORT/WinML backend the pool2d input rank constraint is a RAN…

**Full report, discussion and status: https://github.com/ibelem/chromium/issues/50**

| | |
| --- | --- |
| CWE | CWE-617: Reachable Assertion (CHECK failure → process abort / DoS) |
| Location | `targets/chromium/services/webnn/public/cpp/graph_validation_utils.cc:2276` — `ValidatePool2dAndInferOutput()` |
| Trust boundary | WebNN Pool2d Mojo op from a compromised renderer, validated in the GPU/utility process via WebNNGraphBuilderImpl → OperationValidationContext::ValidatePool2d → ValidatePool2dAndInferOutput. |
| Validated for | chromiumWebnn |

This file is an index entry only — it deliberately does not duplicate the
issue. See https://github.com/ibelem/chromium/issues/50 for the root cause, exploit, reproduction and
suggested fix.


Models: ideaCreator `claude-4-8-opus`, techLead `claude-4-8-opus`, researcher `claude-4-8-opus`, validator `claude-4-8-opus`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #787.