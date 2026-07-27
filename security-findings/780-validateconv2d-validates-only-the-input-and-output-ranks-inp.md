# Security finding #780: ValidateConv2d validates only the *input* and *output* ranks (`inpu…

**Summary:** ValidateConv2d validates only the *input* and *output* ranks (`inpu…

**CWE IDs:** CWE-125: Out-of-bounds Read (missing rank validation, CWE-20/CWE-129)
**Severity / Impact:** A compromised/malicious renderer causes an out-of-bounds heap read inside the privileged WebNN service (GPU/utility process). The leaked adjacent heap word becomes `filter_width` (or `input_channels` for kHwoi), which is then used in the output-size arithmetic and can be reflected back to the renderer through the accept/reject decision of the `validated_output != output->descriptor` check (webnn_graph_builder_impl.cc:1049) — a heap-layout/ASLR oracle (info leak). Where Chromium's libc++ hardening bounds-checks `vector::operator[]`, the same input instead trips a renderer-triggerable abort of the WebNN/GPU process (DoS). The garbage value also flows into graph_builder_ort.cc:1096-1100, which again indexes `filter_shape[3]` when emitting the ONNX ConvTranspose node.
**Affected location:** `targets/chromium/services/webnn/webnn_graph_builder_impl.cc:1004` — `OperationValidationContext::ValidateConv2d()`
**Validated for repos:** chromiumWebnn
**Trust boundary:** Renderer → GPU/utility process WebNN service: `mojom::GraphInfo` operands/operations delivered to `WebNNGraphBuilderImpl::CreateGraph` (untrusted, fully renderer-controlled operand descriptors).

## Description / Root cause
ValidateConv2d validates only the *input* and *output* ranks (`input->descriptor.Rank() != 4 || output->descriptor.Rank() != 4`, webnn_graph_builder_impl.cc:1006) and never checks the rank of the filter operand before handing `filter->descriptor` to ValidateConv2dAndInferOutput / ValidateConvTranspose2dAndInferOutput (lines 1031-1042). Inside those functions the only filter gate is `data_type_limits.conv_transpose2d_input.Supports(filter)` (graph_validation_utils.cc:834) / `conv2d_input.Supports(filter)` (:720), and for the ORT backend that limit is `{DataTypeConstraint::kFloat16To32, {3, 8}}` (ort/context_impl_ort.cc:284,287), i.e. SupportedRanks{min=3,max=8} — a rank-3 filter is accepted. The code then unconditionally indexes four elements of the filter shape vector: graph_validation_utils.cc:863-884 (`filter_shape[0]`…`filter_shape[3]` for kIohw/kOhwi/kHwoi) and :742-773 for the direct path. With a rank-3 filter, `filter_shape[3]` reads one element past the end of the 3-element heap `std::vector<uint32_t>` returned by `OperandDescriptor::shape()` (operand_descriptor.h:140,172). `SupportedRanks::Supports` (supported_tensors.h:44) is the only rank gate and it does not require exactly 4.

**Validator analysis:** CONFIRMED as a real missing-validation defect in the entry repo. Tracing it: `OperationValidationContext::ValidateConv2d` (webnn_graph_builder_impl.cc:987-1054) resolves the three operands, then at :1006 gates only `input->descriptor.Rank() != 4 || output->descriptor.Rank() != 4` — the filter descriptor is passed straight through to `ValidateConv2dAndInferOutput` (:1031) / `ValidateConvTranspose2dAndInferOutput` (:1039). Inside those, the only filter gate is the data-type/rank tensor limit: `conv2d_input.Supports(filter)` (graph_validation_utils.cc:720) and `conv_transpose2d_input.Supports(filter)` (:834). `SupportedTensors::Supports` only tests `ranks.Supports(rank)` with `min <= rank <= max` (supported_tensors.h:44,61-65), and for the ORT context those limits are `{DataTypeConstraint::kFloat16To32, {3, 8}}` (ort/context_impl_ort.cc:284,287) — i.e. min rank 3. Immediately afterwards the code unconditionally reads four shape elements: for the ORT context the input layout is kNchw, so `ConvertToConv2dAttributes` selects `Conv2dFilterOperandLayout::kOihw` (webnn_graph_builder_impl.cc:186-190) and graph_validation_utils.cc:767-773 executes `filter_width = filter_shape[3]` on a 3-element vector; the transposed path is equally exposed at :863-884. The indexing occurs BEFORE any groups/channel consistency check (:777-791, :886-893), so nothing rejects the short filter first, and there is no try/catch — `std::vector::operator[]` throws nothing. vulnType CWE-125 (root cause CWE-20/CWE-129, missing rank validation) is accurate. The IMPACT claim is partly overstated: in Chrome builds with libc++ hardening (fast mode checks `vector::operator[]` element access) this is a deterministic hard CHECK/trap, i.e. a compromised-renderer-triggerable crash of the privileged GPU/utility process rather than a usable heap oracle; only in an unhardened build does it become the 4-byte adjacent-word read / accept-reject oracle described. Either way it is a renderer-crossing memory-safety defect in privileged code, not a resource-exhaustion issue, so it is in scope. The proposedFix is correct and sufficient in its primary form: extending webnn_graph_builder_impl.cc:1006 to also require `filter->descriptor.Rank() == 4` closes the only reachable path, and the suggested defense-in-depth `filter.Rank() != 4` early-return in both graph_validation_utils.cc functions (right after the `Supports(filter)` checks at :725 and :841) is the more robust placement because those helpers are also called from the other backends and would otherwise stay indexable at :748-772/:865-882. I would prefer the graph_validation_utils.cc hardening as the primary fix (it protects every caller and every layout) plus the builder-side check. Tightening `conv2d_input`/`conv_transpose2d_input` to `SupportedRanks::Exactly(4)` (ort/context_impl_ort.cc:284,287) is also defensible — WebNN conv2d is 4-D by spec — but it additionally changes the `opSupportLimits()` values reported to script, so it should not be the only remedy.

## Exploit / Proof of Concept
From a compromised renderer, send a `GraphInfo` over the WebNN Mojo pipe containing a `Conv2d` operation with `kind = kTransposed`, a rank-4 float32 input operand (e.g. [1,1,4,4]), a rank-4 output operand (so both rank==4 checks at line 1006 pass), and a **rank-3** float32 filter operand, e.g. shape [1,1,3]. `OperandDescriptor::Create` accepts it (rank ≤ 8, all dims non-zero), `conv_transpose2d_input.Supports(filter)` accepts rank 3, and execution reaches graph_validation_utils.cc:868 (`filter_width = filter_shape[3]`), reading past the end of the 3-element shape vector. Nothing on the path rejects the rank-3 filter first, and no try/catch is involved (this is a plain vector index, not an exception). By varying the declared output spatial dims and observing whether CreateGraph succeeds, the renderer binary-searches the out-of-bounds word.

## Reproduction (steps)
```
Not web-reachable as a plain HTML page — repro requires a compromised renderer speaking WebNN Mojo directly.

Why no HTML repro: the rank-3 filter is rejected in the renderer before it ever reaches the service. Blink's MLGraphBuilder validates the conv2d/convTranspose2d filter operand rank (4-D required) and throws a TypeError, so no sequence of `navigator.ml` / `MLGraphBuilder` calls from web content can put a rank-3 filter into `mojom::GraphInfo`. The cited code is only reachable once the renderer-side check is bypassed, which is exactly the WebNN trust boundary (renderer -> GPU/utility process) this scan assumes.

Concrete repro steps (compromised-renderer / fuzz-harness level):
1. Build Chromium with the WebNN ORT backend enabled (`--enable-features=WebMachineLearningNeuralNetwork`, ORT/OpenVINO EP context so that `ContextImplOrt::GetProperties` supplies `conv2d_input = {kFloat16To32, {3,8}}`, ort/context_impl_ort.cc:284).
2. From a renderer-side harness that writes `webnn::mojom::GraphInfo` directly onto the `WebNNGraphBuilder` pipe (e.g. a mojo-js / fuzzer harness, bypassing blink/renderer/modules/ml validation), build a GraphInfo with three operands:
   - operand 0: input, kFloat32, shape [1,1,4,4]  (rank 4 -> passes webnn_graph_builder_impl.cc:1006)
   - operand 1: filter, kFloat32, shape [1,1,3]   (rank 3 -> accepted by OperandDescriptor::Create and by conv2d_input.Supports)
   - operand 2: output, kFloat32, shape [1,1,2,2]  (rank 4 -> passes :1006)
3. Add one `mojom::Conv2d` operation: `kind = kDirect` (or `kTransposed`), input_operand_id = 0, filter_operand_id = 1, output_operand_id = 2, groups = 1, strides/dilations = {1,1}, padding all zeros, no bias.
4. Call `CreateGraph`. Validation reaches `ValidateConv2dAndInferOutput` -> graph_validation_utils.cc:767-773 (`kOihw`: `filter_width = filter_shape[3]`) with a 3-element shape vector.
5. Observed result: with ASan/libc++ hardening, a `container-overflow`/`heap-buffer-overflow` READ of 4 bytes in `webnn::ValidateConv2dAndInferOutput` (or a libc++ hardening `__libcpp_verbose_abort` on `vector::operator[]`) in the GPU/utility process. Without hardening, `CreateGraph` succeeds or fails depending on the adjacent heap word, and the renderer can binary-search that word by varying the declared output spatial dims (the `validated_output != output->descriptor` comparison at webnn_graph_builder_impl.cc:1049).
6. The same rank-3 filter also reaches the ORT graph emitter's `filter_shape[3]` use in services/webnn/ort/graph_builder_ort.cc (~:1096-1100) if validation is passed.
```

## Test
_(not provided)_


## Suggested fix
Validate the filter rank against the input rank/layout before converting attributes. In `OperationValidationContext::ValidateConv2d`, extend the check at webnn_graph_builder_impl.cc:1006 to `if (input->descriptor.Rank() != 4 || output->descriptor.Rank() != 4 || filter->descriptor.Rank() != 4) return false;`. Additionally harden the sinks so they cannot be reached with a short shape: in `ValidateConv2dAndInferOutput` and `ValidateConvTranspose2dAndInferOutput`, add `if (filter.Rank() != 4) return base::unexpected(ErrorWithLabel(label, "The filter should be a 4-D tensor."));` immediately after the `Supports(filter)` data-type check (graph_validation_utils.cc:725 and :841), or tighten `conv2d_input`/`conv_transpose2d_input` in ort/context_impl_ort.cc:284,287 to `SupportedRanks::Exactly(4)`.

## Models used

| Role | Provider / model |
| --- | --- |
| Idea | `claude-4-8-opus` |
| Tech lead | `claude-4-8-opus` |
| Researcher | `claude-5-opus` |
| Validator | `claude-5-opus` |


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #780.
