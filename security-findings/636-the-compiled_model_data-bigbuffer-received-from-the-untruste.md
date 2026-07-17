# Security finding #636: The `compiled_model_data` BigBuffer received from the untrusted Com…

**Summary:** The `compiled_model_data` BigBuffer received from the untrusted Com…

**CWE IDs:** CWE-367: Time-of-check Time-of-use (TOCTOU) Race Condition
**Severity / Impact:** A compromised Compiler process can exploit the TOCTOU window: it sends a large buffer (>64KB) that uses shared memory backing, then modifies the shared memory region while ORT's protobuf parser is reading it in the GPU process. This can cause the parser to read inconsistent data, potentially triggering memory corruption bugs in the protobuf parser or downstream ONNX model processing. This affects any Chromium WebNN user on Windows whose browser uses the ORT backend, as the GPU process runs with elevated privileges compared to the sandboxed Compiler process.
**Affected location:** `targets/chromium/services/webnn/ort/graph_impl_ort.cc:214` — `GraphImplOrt::CreateSessionFromCompiledGraph()`
**Validated for repos:** chromiumWebnn
**Trust boundary:** WebNNModelLoader.LoadCompiledGraph Mojo IPC from sandboxed Compiler process to GPU process

## Description / Root cause
The `compiled_model_data` BigBuffer received from the untrusted Compiler process is passed directly to ORT's protobuf parser without making a defensive copy. BigBuffer's documentation (targets/chromium/mojo/public/cpp/base/big_buffer.h:74-78) explicitly warns: 'When shmem is backing the message, it may be writable in the sending process while being read in the receiving process. If a BigBuffer is received from an untrustworthy process, you should make a copy of the data before processing it to avoid time-of-check time-of-use (TOCTOU) bugs.' At line 214, the BigBuffer is passed by value (move semantics), but its backing storage (shared memory region for buffers >64KB per big_buffer.h:85) remains mapped and potentially writable by the compromised Compiler process. The data() pointer (line 222) is passed to ort_api->CreateSessionFromArray, which invokes protobuf parsing without copying first.

**Validator analysis:** The reported TOCTOU is real and accurately categorised as CWE-367. big_buffer.h:74-78 carries an explicit SECURITY NOTE that a BigBuffer received from an untrustworthy process must be copied before processing because kMaxInlineBytes=64KB (big_buffer.h:85) is the inline threshold and larger buffers are shmem-backed and sender-mutable. GraphImplOrt::CreateSessionFromCompiledGraph at graph_impl_ort.cc:214-223 takes the BigBuffer by value (move only, no defensive copy) and hands data() directly to ort_api->CreateSessionFromArray, creating exactly the window cited. Impact (parser sees inconsistent bytes during iteration → memcorrupt/crash in privileged GPU process) is plausible given protobuf parsers do not assume concurrent mutation of their input. Proposed fix is correct and sufficient: either memcpy into a std::unique_ptr<uint8_t[]> before parsing, or use BigBuffer::Clone (big_buffer.cc:108) which constructs via base::span, forcing a fresh heap-backed copy. Minor refinement: also assert compiled_model_data.size() > 0 and consider pinning/copying before any other read of .data() downstream. The finding is correctly scoped to the entry surface (services/webnn is part of chromiumWebnn), so downstream repos are all NA.

## Exploit / Proof of Concept
1) Attacker compromises the Compiler process (or it's already untrusted by design). 2) Compiler creates a BigBuffer >64KB with valid-looking ONNX protobuf data, forcing shared memory backing. 3) Before ORT parsing completes, Compiler modifies the shared memory to corrupt protobuf fields (e.g., change a repeated field count from 1 to 0x7FFFFFFF while protobuf is iterating). 4) ORT's protobuf parser reads the modified value during iteration, causing out-of-bounds read or buffer over-read leading to crash or potential code execution in GPU process.

## Reproduction (steps)
```
1. Launch Chromium with --enable-features=WebMachineLearningNeuralNetwork on Windows where the ORT backend is active. 2. From a renderer, call navigator.ml.createContext() and then MLGraphBuilder.build() with a graph whose serialized ONNX output would exceed 64KB (large weight constants to push the Mojo BigBuffer of the compiled graph past kMaxInlineBytes). 3. Simultaneously compromise/debugger-attach the WebNN Compiler (service) process and, after the Mojo IPC that sends LoadCompiledGraph's BigBuffer is dispatched but before ORT's CreateSessionFromArray returns in the GPU process, mutate the shmem region backing the BigBuffer (e.g. flip a repeated-field count). 4. Observe crash / sanitizer abort / inconsistent parse in the GPU process. Note: step 3 requires control of the sandboxed Compiler process, which is not reachable from a pure web page — this reproduces the TOCTOU only under an attacker-model where the Compiler process is already compromised, matching the reported trust boundary.
```

## Test
_(not provided)_


## Suggested fix
In graph_impl_ort.cc:CreateSessionFromCompiledGraph (line 221), before calling ort_api->CreateSessionFromArray, make a defensive copy of the BigBuffer data to break the shared memory aliasing: `auto model_data_copy = std::make_unique<uint8_t[]>(compiled_model_data.size()); std::memcpy(model_data_copy.get(), compiled_model_data.data(), compiled_model_data.size());` then pass `model_data_copy.get()` and `compiled_model_data.size()` to CreateSessionFromArray. Alternatively, use `compiled_model_data.Clone()` which forces a copy via BigBuffer(base::span(*this)) per big_buffer.cc:108, ensuring the copy uses fresh memory allocation independent of the IPC shared memory region.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #636.
