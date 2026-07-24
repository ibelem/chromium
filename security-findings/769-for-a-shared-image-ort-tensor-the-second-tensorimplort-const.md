# Security finding #769: For a shared-image ORT tensor, the second TensorImplOrt constructor…

**Summary:** For a shared-image ORT tensor, the second TensorImplOrt constructor…

**CWE IDs:** CWE-617: Reachable Assertion
**Severity / Impact:** A compromised renderer can deterministically crash the GPU/utility process hosting the WebNN service (denial of service). The CHECK is a controlled abort (no memory corruption), but it is unconditionally reachable across the Mojo trust boundary.
**Affected location:** `targets/chromium/services/webnn/ort/tensor_impl_ort.cc:74` — `TensorImplOrt::ReadTensorImpl()`
**Validated for repos:** chromiumWebnn
**Trust boundary:** Compromised renderer → WebNN Mojo (WebNNTensor::ReadTensor) → GPU/utility process ReadTensorImpl

## Description / Root cause
For a shared-image ORT tensor, the second TensorImplOrt constructor (tensor_impl_ort.cc:37-52) stores size_ = buffer_size, where buffer_size = representation->size().width() (context_impl_ort.cc:563-564). CreateTensorFromSharedImageImpl only validates buffer_size >= PackedByteLength() (context_impl_ort.cc:565), and the comment at 561-562 explicitly states the shared-image buffer may be LARGER than required due to D3D12 heap alignment padding. AsSpan() (line 56-68) then returns a span of size_ (== buffer_size) bytes. ReadTensorImpl at line 74 does CHECK_EQ(PackedByteLength(), buffer_span.size()); whenever the shared image is alignment-padded so buffer_size > PackedByteLength(), the equality fails and the process aborts.

**Validator analysis:** The vulnType (CWE-617 reachable assertion) and impact (renderer-triggered controlled abort of the GPU/utility process, no memory corruption) are accurate and IN SCOPE — CWE-617 is explicitly carved back in by the base out-of-scope block (it is a specific in-scope crash class, distinct from CWE-400/770/789 resource exhaustion). Data flow is confirmed: CreateTensorFromMailbox (webnn_context_impl.cc:427) → CreateTensorFromSharedImageImpl stores size_=buffer_size=representation->size().width() (context_impl_ort.cc:563-564,656-659) with only a >= check (565) despite the padding comment (561-562); the CPU-mapped fallback sets can_access_on_cpu=true (643) so a kRead shared-image tensor is NOT rejected at 646-652; ReadTensor's kRead usage guard (webnn_tensor_impl.cc:62) then passes and schedules ReadTensorImpl, whose CHECK_EQ(PackedByteLength(), buffer_span.size()) at tensor_impl_ort.cc:74 aborts whenever alignment padding makes buffer_size>PackedByteLength(). The proposed fix (CHECK_GE + buffer_span.first(PackedByteLength())) is correct for the cited read line and also avoids leaking padding bytes, but it is INCOMPLETE: WriteTensorImpl (tensor_impl_ort.cc:82) has the identical latent abort — ReadDataFromBigBufferOrDataPipe does dst_span.copy_from(src_buffer) (webnn_context_impl.cc:413) with dst_span=AsSpan()=buffer_size and src_buffer validated to PackedByteLength (webnn_tensor_impl.cc:100), so span::copy_from's size-equality CHECK will also fire. A more robust fix is to make AsSpan()/the logical accessors clamp to PackedByteLength() (or store the logical length separately) so both read and write paths only ever touch the tensor's packed bytes. Reachability caveat: the divergence depends on D3D12 heap/placement alignment padding of the WebGPU-interop shared image, which the renderer influences but cannot deterministically force in every environment; nonetheless the developers' own comment and >= check confirm the padded case is expected, making the CHECK a real reachable abort.

## Exploit / Proof of Concept
Renderer creates a WebGPU-interop shared-image tensor with usage kRead. The kRead flag is permitted on a shared-image tensor: in the mapped-buffer fallback path (context_impl_ort.cc:623-644) can_access_on_cpu is set true, so the kRead/kWrite rejection at 646-652 does not fire, and the CreateTensorWithDataAsOrtValue call is made with the full buffer_size. If the underlying D3D12 shared buffer is padded to an alignment boundary (e.g. 64KB placement alignment) so representation->size().width() > descriptor.PackedByteLength(), size_ > PackedByteLength(). The renderer then calls ReadTensor() (usage guard at webnn_tensor_impl.cc:62 passes because kRead is set), which schedules ReadTensorImpl on the GPU sequence; CHECK_EQ(PackedByteLength(), buffer_span.size()) at line 74 fails and aborts the process.

## Reproduction (steps)
```
Web-reachable-but-not-self-contained: the abort requires a WebGPU→WebNN interop MLTensor whose backing D3D12 shared buffer is alignment-padded so representation->size().width() > descriptor.PackedByteLength(). This cannot be forced deterministically from a pure open-in-browser WebNN page because the renderer does not directly control D3D12 heap placement/64KB alignment padding, so an HTML page that merely calls the API is not a guaranteed trigger. Steps: (1) Launch Chrome with --enable-features=WebMachineLearningNeuralNetwork,WebGPU on a Windows/D3D12 build where the ORT external_resource_importer is unavailable (forcing the CPU-mapped fallback at context_impl_ort.cc:623-644). (2) From a (compromised) renderer, create an MLContext, obtain a GPUBuffer via WebGPU, and import it as an MLTensor with usage {webgpuInterop:true, read:true} using a small descriptor (e.g. int8 shape [1]) whose PackedByteLength() (1 byte) is far below the heap placement alignment (64KB), so the produced shared image size is padded. (3) Call tensor.readTensor() (MLTensorUsageFlags::kRead passes webnn_tensor_impl.cc:62). (4) On the GPU/WebNN sequence, TensorImplOrt::ReadTensorImpl runs CHECK_EQ(PackedByteLength()=1, buffer_span.size()=paddedSize) at tensor_impl_ort.cc:74, which fails and aborts (crashes) the GPU/utility process. To make the trigger deterministic for testing, instrument/force the shared-image producer to return a size larger than PackedByteLength().
```

## Test
_(not provided)_


## Suggested fix
Do not CHECK_EQ against the possibly-padded span. Size the returned data to the logical packed length: replace lines 73-76 with `base::span<const uint8_t> buffer_span = AsSpan(); CHECK_GE(buffer_span.size(), PackedByteLength()); buffer_span = buffer_span.first(PackedByteLength());` before writing to the data pipe / BigBuffer. This tolerates alignment-padded shared-image buffers while still guarding against undersized buffers, and only ever exposes the tensor's actual PackedByteLength() bytes to the renderer (also avoiding leaking padding bytes).

## Models used

| Role | Provider / model |
| --- | --- |
| Idea | `claude-4-6-sonnet` |
| Tech lead | `claude-4-6-sonnet` |
| Researcher | `claude-4-8-opus` |
| Validator | `claude-4-8-opus` |


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #769.
