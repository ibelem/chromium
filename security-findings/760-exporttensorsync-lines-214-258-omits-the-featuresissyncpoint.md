# Security finding #760: ExportTensorSync (lines 214-258) omits the `features::IsSyncPointGr…

**Summary:** ExportTensorSync (lines 214-258) omits the `features::IsSyncPointGr…

**CWE IDs:** CWE-367: Time-of-check Time-of-use (TOCTOU) Race Condition
**Severity / Impact:** A compromised renderer can create a window where both WebNN (still holding `representation_access_`) and WebGPU hold concurrent access to the same shared image — the exact cross-process release-token race the ExportTensor guard was added to prevent. WebGPU commands on their own sequence are not serialized with WebNN's `sequence_id_` and, without a release fence, are not ordered after the pending ExportTensorImpl. Result: data race / use-after-export on shared GPU tensor memory in the GPU process, i.e. memory corruption reachable from untrusted web content (sandbox-escape class).
**Affected location:** `targets/chromium/services/webnn/webnn_tensor_impl.cc:214` — `WebNNTensorImpl::ExportTensorSync()`
**Validated for repos:** chromiumWebnn
**Trust boundary:** compromised renderer → GPU/utility process via the Mojo WebNNTensor interface; the renderer-supplied `release_count` argument and the shared-image tensor exported to WebGPU cross the boundary

## Description / Root cause
ExportTensorSync (lines 214-258) omits the `features::IsSyncPointGraphValidationEnabled()` guard that its async twin ExportTensor enforces at lines 170-173. That guard exists specifically "to avoid a cross-process release-token race on the shared-image tensor memory" (comment at 168-169). ExportTensorSync instead: (1) builds a `release` gpu::SyncToken only when `release_count != 0` (guard at 229-234), so a renderer-supplied `release_count=0` leaves `release` default-constructed; (2) passes that empty release to `RunOrScheduleTask(..., {}, release)` (line 255), so the gpu::Scheduler signals no fence when the export task later runs `ExportTensorImpl(std::move(self->representation_access_))` on the GPU sequence (line 251); and (3) fires `std::move(callback).Run()` synchronously on the Mojo thread (line 257) before that GPU-sequence task has executed. The WebNN access (`representation_access_`) is thus still live on the GPU sequence when the renderer regains control, and no SyncToken gates WebGPU consumers on a separate command-buffer sequence.

**Validator analysis:** The vuln type (CWE-367 TOCTOU) is accurate. ExportTensor (lines 165-212) has a hard guard at lines 170-173 that rejects calls when IsSyncPointGraphValidationEnabled() is false, precisely to prevent a cross-process release-token race (comment at lines 168-169). ExportTensorSync (lines 214-258) is documented in the mojom (webnn_tensor.mojom:72-76) as the synchronous fallback for when that feature is disabled, but it applies no equivalent gate. The critical sequence is: RunOrScheduleTask() posts the ExportTensorImpl work to the GPU sequence (line 236-255), then immediately — without waiting for that work to run — fires std::move(callback).Run() at line 257. Since the Mojo [Sync] reply goes back to the renderer at that point, the renderer unblocks and can submit WebGPU commands against the shared image before ExportTensorImpl has executed and dropped representation_access_. With release_count==0 no SyncToken release is built (lines 229-234), so WebGPU consumers have no fence to wait on. This is a real concurrent-access window on the shared GPU image. The impact claim (data race / use-after-export = memory corruption reachable from compromised renderer) is accurate — concurrent GPU-process access to the same shared image without ordering fences is a genuine GPU-memory-corruption-class bug. The proposed fix is correct and sufficient: adding the IsSyncPointGraphValidationEnabled() guard at the top of ExportTensorSync and rejecting release_count==0 for kWebGpuInterop tensors are together the minimal safe fix; alternatively, moving callback.Run() to inside the GPU-sequence closure (after ExportTensorImpl completes) would also close the race. The flaw is reachable from a compromised renderer via the Mojo WebNNTensor::ExportTensorSync message with a renderer-controlled release_count=0.

## Exploit / Proof of Concept
From a compromised renderer with a kWebGpuInterop tensor: call ExportTensorSync(flow_id, release_count=0, callback). The synchronous callback returns immediately while the ExportTensorImpl task is still queued on the GPU sequence. Because release_count==0, no SyncToken is signaled, so the renderer can immediately submit WebGPU work that begins accessing the shared image before WebNN's `representation_access_` scoped access is dropped — producing concurrent access to the shared image. The async ExportTensor path would refuse this when the sync-point feature is disabled; ExportTensorSync has no equivalent gate and no requirement that release_count be non-zero.

## Reproduction (steps)
```
The triggering Mojo call (ExportTensorSync with release_count=0) is an internal IPC message not directly exposable from normal JavaScript. A normal web page cannot craft arbitrary Mojo messages — only a compromised renderer process can. Therefore no self-contained HTML repro is possible.

Concrete repro steps (requires a compromised renderer / Mojo fuzzer):
1. In a Chromium build with ASan + GPU-process-in-process (--in-process-gpu makes the race easier to observe), enable WebNN and WebGPU: --enable-features=WebMachineLearningNeuralNetwork,WebGPU.
2. From the renderer (or a Mojo fuzzer targeting the WebNNTensor interface), create a WebNN tensor with usage kWebGpuInterop.
3. Send the Mojo message WebNNTensor::ExportTensorSync(flow_id=<any>, release_count=0). The [Sync] call returns immediately once the GPU process fires callback.Run() at line 257 of webnn_tensor_impl.cc, BEFORE the GPU-sequence task at line 251 runs ExportTensorImpl.
4. Immediately after the sync call returns (renderer unblocked), submit a WebGPU command that reads/writes the shared image that was just 'exported'.
5. Observe: the GPU sequence later runs ExportTensorImpl, dropping representation_access_, while WebGPU commands are concurrently accessing the same shared image — a data race detectable under ThreadSanitizer or via GPU debug layers as concurrent shared-image access without ordering.
Why not web-reachable: the release_count argument is an internal Mojo parameter chosen by the browser-side WebNN JS binding, not directly settable by untrusted JavaScript. Exploiting this requires a compromised renderer that can craft Mojo IPC directly.
```

## Test
_(not provided)_


## Suggested fix
Give ExportTensorSync the same protection as ExportTensor: (1) add `if (!features::IsSyncPointGraphValidationEnabled()) { GetMojoReceiver().ReportBadMessage(kBadMessageAsyncExportNotSupported); return; }` at the top; and (2) reject `release_count == 0` for kWebGpuInterop exports (e.g. `if (release_count == 0) { ReportBadMessage(kBadMessageInvalidTensor); return; }`) so a real release SyncToken is always built and the scheduler signals a fence that WebGPU consumers must wait on. Alternatively, defer running `callback` until the GPU-sequence export task has actually executed (post it from inside the scheduled task) so the renderer cannot observe completion before representation_access_ is released.

## Models used

| Role | Provider / model |
| --- | --- |
| Idea | `claude-4-6-sonnet` |
| Tech lead | `claude-4-6-sonnet` |
| Researcher | `claude-4-8-opus` |
| Validator | `claude-4-6-sonnet` |


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #760.
