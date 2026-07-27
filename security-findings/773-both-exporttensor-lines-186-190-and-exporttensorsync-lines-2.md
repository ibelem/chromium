# Security finding #773: Both `ExportTensor` (lines 186-190) and `ExportTensorSync` (lines 2…

**Summary:** Both `ExportTensor` (lines 186-190) and `ExportTensorSync` (lines 2…

**CWE IDs:** CWE-416: Use After Free via unvalidated release-fence count
**Severity / Impact:** WebGPU, waiting on that release fence, begins writing to the shared image while `representation_access_` is still being destroyed on a separate sequence (see DestroyAccessAndRepresentationAndWait / OnTaskRunnerDeleter, lines 315-347), producing a use-after-free / concurrent access on the shared-image backing memory in the GPU process — memory corruption reachable from a compromised renderer (sandbox-escape class).
**Affected location:** `targets/chromium/services/webnn/webnn_tensor_impl.cc:229` — `WebNNTensorImpl::ExportTensorSync()`
**Validated for repos:** chromiumWebnn
**Trust boundary:** renderer → GPU/utility-process WebNN service over Mojo (`release_count` uint64_t) → gpu::Scheduler release fence

## Description / Root cause
Both `ExportTensor` (lines 186-190) and `ExportTensorSync` (lines 230-234) build `gpu::SyncToken release` directly from the renderer-supplied `release_count` with no check that it is strictly greater than the WebNN command buffer's current release counter. The token is handed to `GpuTaskScheduler::ScheduleGpuTaskImpl` (gpu_task_scheduler.cc:46-63) as the `release` argument of `gpu::Scheduler::Task`. A stale / already-consumed `release_count` is signaled by the scheduler immediately (or out of order), so the WebNN sequence's release fence fires before `ExportTensorImpl(std::move(representation_access_))` (line 251) has finished tearing down the shared-image access on its deleter-bound sequence.

**Validator analysis:** The cited code is real and matches the report: both ExportTensor (185-190) and ExportTensorSync (229-234) construct gpu::SyncToken(namespace_id, command_buffer_id, release_count) from a raw renderer-controlled uint64_t Mojo field and pass it as the scheduler Task's `release` (gpu_task_scheduler.cc:46-63) with NO check that release_count strictly exceeds the WebNN command buffer's last released count. Only namespace_id/command_buffer_id come from the trusted WebNN command buffer; release_count crosses the renderer→GPU Mojo boundary unchecked. A command-buffer decoder normally validates strictly-increasing fence releases; WebNN bypasses that. A stale/already-consumed release_count means any WebGPU-side wait on that count is already satisfied, so WebGPU's queued shared-image access proceeds without actually waiting for ExportTensorImpl (line 251) to finish ending WebNN's scoped access — concurrent BeginAccess/EndAccess on the shared image, a synchronization/UAF-class hazard in the GPU process reachable from a compromised renderer. The developers' own comment at line 168-169 ('cross-process release-token race on the shared-image tensor memory') confirms the hazard class, yet ExportTensorSync lacks even the feature-flag guard applied to ExportTensor. vulnType CWE-416/race and the sandbox-escape-class impact are accurate (this is memory-safety, not DoS, so in scope). The proposed fix is correct in direction — validate release_count is strictly greater than the last-issued release count on the WebNN sequence's command buffer and ReportBadMessage otherwise, in BOTH paths — and ExportTensor should keep its feature gate. This closes the boundary at the exact point the untrusted value is consumed. Downstream repos (ORT/OVEP/OpenVINO) are unrelated to this GPU-process interop path. Note: release_count is an internal WebGPU-interop Mojo field, not settable from JS WebNN, so a self-contained web-reachable HTML repro is not possible; the trust boundary is explicitly a compromised renderer forging Mojo — hence steps rather than an HTML page, and no WPT (non-deterministic race, not cleanly assertable). No unit-test kind applies to chromiumWebnn (its kinds are wpt+html).

## Exploit / Proof of Concept
Renderer exports a `kWebGpuInterop` tensor and passes a `release_count` that is <= the command buffer's current release counter (e.g. reuse an already-consumed value, or 1). The scheduler signals the release fence immediately; WebGPU's queued access to the shared image proceeds concurrently with the still-in-flight teardown of `representation_access_`, racing on freed/reclaimed image memory.

## Reproduction (steps)
```
Not web-reachable via pure JS: release_count is an internal WebNN↔WebGPU interop Mojo parameter set by the browser's interop plumbing, not exposed to page script. Repro requires a compromised renderer (the stated trust boundary) forging Mojo messages. Steps: (1) Build Chromium with WebNN + WebGPU interop enabled (kWebGpuInterop tensor usage) and the SyncPointGraphValidation feature on. (2) In an instrumented/compromised renderer, create an MLContext and an MLTensor with kWebGpuInterop usage; import it into WebGPU so the shared image and WebNN command-buffer sync sequence exist. (3) Drive one legitimate export so the WebNN sequence's release counter advances to some value C (>0). (4) Forge a WebNNTensor::ExportTensorSync (or ExportTensor) Mojo call carrying release_count = a value <= C (e.g. reuse C, or send 1). WebNNTensorImpl::ExportTensorSync (webnn_tensor_impl.cc:229-234) builds gpu::SyncToken with that stale count and schedules the export task with it as the release fence. (5) Queue a WebGPU access to the same shared image that waits on SyncToken(webnn_namespace, cb_id, <=C); because that count was already released, the wait is pre-satisfied and WebGPU's access runs concurrently with the still-in-flight teardown of representation_access_ (ExportTensorImpl at line 251 / OnTaskRunnerDeleter teardown at 315-347). Observe under TSan/ASan a data race / use-after on the shared-image access state. Expected after fix: the service rejects the call via ReportBadMessage because release_count is not strictly greater than the last released count.
```

## Test
_(not provided)_


## Suggested fix
Validate `release_count` against the command buffer's current release counter before constructing the SyncToken in BOTH paths: reject (ReportBadMessage) if `release_count` is not strictly greater than the last released count on `context_->gpu_task_scheduler()`'s command buffer (mirroring the validation gpu command decoders apply to fence releases). Track the last-issued release count per WebNN sequence and require strict monotonic increase.

## Models used

| Role | Provider / model |
| --- | --- |
| Idea | `claude-4-6-sonnet` |
| Tech lead | `claude-4-6-sonnet` |
| Researcher | `claude-4-8-opus` |
| Validator | `claude-4-8-opus` |


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #773.
