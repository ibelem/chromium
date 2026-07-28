# Security finding #788: For the ORT backend, `matmul_input` is configured as `kMaxRank = Su…

**Full report, discussion and status: https://github.com/ibelem/chromium/issues/53**

| | |
| --- | --- |
| CWE | CWE-617: Reachable Assertion (renderer-triggerable CHECK abort) |
| Location | `targets/chromium/services/webnn/public/cpp/graph_validation_utils.cc:2126` — `ValidateMatmulAndInferOutput()` |
| Trust boundary | WebNN Mojo OperandDescriptor deserialization from a compromised renderer (operand_descriptor_mojom_traits.cc StructTraits::Read → CreateForDeserialization) |
| Validated for | chromiumWebnn |

This file is an index entry only — it deliberately does not duplicate the
issue. See https://github.com/ibelem/chromium/issues/53 for the root cause, exploit, reproduction and
suggested fix.


Models: ideaCreator `claude-4-8-opus`, techLead `claude-4-8-opus`, researcher `claude-4-8-opus`, validator `claude-4-8-opus`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #788.