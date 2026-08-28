# Security finding #990: Tensor disconnection synchronously destroys shared-image access and…

**Full report, discussion and status: https://github.com/ibelem/chromium/issues/75**

| | |
| --- | --- |
| CWE | CWE-672: Operation on a Resource after Expiration or Release |
| Location | [`targets/chromium/services/webnn/webnn_context_impl.cc:615`](https://github.com/chromium/chromium/blob/4ba63ef5580ad0663edf04127811a534ecec78e4/services/webnn/webnn_context_impl.cc#L615-L626) — `WebNNContextImpl::RemoveWebNNTensorImpl()` |
| Trust boundary | Compromised WebNN compiler supplies the executable graph while web content controls shared-image tensor endpoint lifetime across Mojo into the GPU process. |
| Validated for | chromiumWebnn |
| Reachable from | chromiumWebnn |

This file is an index entry only — it deliberately does not duplicate the
issue. See https://github.com/ibelem/chromium/issues/75 for the root cause, exploit, reproduction and
suggested fix.


Models: ideaCreator `openai,gpt-5.6-cyber`, techLead `openai,gpt-5.6-cyber`, researcher `openai,gpt-5.6-cyber`, validator `openai,gpt-5.6-cyber`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #990.