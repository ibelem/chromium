# Security finding #637: The compiled_model_data buffer received from the untrusted Compiler…

**Summary:** The compiled_model_data buffer received from the untrusted Compiler…

**CWE IDs:** CWE-502: Deserialization of Untrusted Data
**Severity / Impact:** A compromised Compiler process can craft malicious protobuf data that triggers memory corruption bugs in ORT's protobuf parser or downstream ONNX model loading code (inference_session.cc:1287 calls ParseFromArray on untrusted data). While protobuf parsers are generally robust, they are complex and have historically contained vulnerabilities. The GPU process has higher privileges than the sandboxed Compiler process, making this a privilege escalation vector. This affects Chromium WebNN users on Windows with ORT backend.
**Affected location:** `targets/chromium/services/webnn/ort/graph_impl_ort.cc:221` — `GraphImplOrt::CreateSessionFromCompiledGraph()`
**Validated for repos:** chromiumWebnn
**Trust boundary:** WebNNModelLoader.LoadCompiledGraph Mojo IPC from sandboxed Compiler process to GPU process

## Description / Root cause
The compiled_model_data buffer received from the untrusted Compiler process is passed directly to ORT's ONNX protobuf parser with no validation of format or structure. While ORT performs basic checks (null pointer and size < 2GB in targets/onnxruntime/onnxruntime/core/session/utils.cc:273-275), there is no validation that: (a) the buffer contains valid ONNX protobuf magic bytes, (b) the buffer meets minimum size requirements for a valid ONNX model, (c) the protobuf structure is well-formed before invoking the full parser. The Compiler side checks (compiler_context_impl_ort.cc:178-179) only verify the buffer is non-null and non-zero size. A compromised Compiler can send arbitrary bytes that will be interpreted as an ONNX protobuf, exercising the protobuf parser's error-handling paths which may have memory-safety bugs.

**Validator analysis:** The data flow is confirmed: a sandboxed Compiler process sends compiled_model_data via the WebNNModelLoader.LoadCompiledGraph Mojo IPC (dispatch_context_impl_ort.cc:153-195) to the GPU process. At dispatch_context_impl_ort.cc:192 the buffer is passed directly to GraphImplOrt::CreateSessionFromCompiledGraph, which at graph_impl_ort.cc:221-223 calls ort_api->CreateSessionFromArray with no validation of format, structure, or magic bytes. ORT's utils.cc:273-275 only checks for null pointer and size > 2GB, then inference_session.cc:1287 calls model_proto.ParseFromArray on the raw untrusted bytes. This is a genuine CWE-502 (Deserialization of Untrusted Data) at the trust boundary. The vuln_type is accurate. However, the impact is overstated: the claim of RCE in the GPU process is speculative and depends on hypothetical, unproven bugs in protobuf or ORT model loading. Protobuf's ParseFromArray is extensively fuzzed and returns a clean error (inference_session.cc:1289-1291) on malformed input. The realistic impact is expanded attack surface against the GPU process's deserialization code, which is a legitimate defense-in-depth concern but not a proven memory-safety vulnerability. The proposed fix is partially incorrect: the suggested 'magic bytes' check (data[0] != 0x08 for ONNX ir_version) is wrong because ONNX is standard protobuf with no magic bytes — a valid model need not start with field tag 0x08 (e.g., if other fields come first). The minimum-size check is reasonable. A better fix would be: (a) add a minimum size check (e.g., >= 2 bytes) at dispatch_context_impl_ort.cc before calling CreateSessionFromCompiledGraph, (b) after ParseFromArray succeeds, validate that the ModelProto has required ONNX fields (ir_version, graph.opset_ir_version, graph.input, graph.output) before proceeding, and (c) use protobuf's CodedInputStream with SetTotalBytesLimit and recursion depth limits to harden against resource exhaustion. The finding is correctly categorized as CWE-502 and the trust boundary (Compiler→GPU Mojo IPC) is correctly identified.

## Exploit / Proof of Concept
1) Attacker compromises the Compiler process. 2) Compiler sends a crafted buffer that passes the non-null and non-zero checks but contains malformed protobuf data designed to trigger parser bugs (e.g., deeply nested messages to exhaust stack, malformed varints to cause integer overflow in length calculations, or carefully crafted field tags that exploit parser edge cases). 3) ORT's ParseFromArray or subsequent model loading code encounters the malformed data and triggers a memory-safety violation. 4) If the parser has a heap-buffer-overflow or use-after-free bug in error handling, this becomes RCE in the GPU process.

## Reproduction (html)
```
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>WebNN CWE-502 Repro — untrusted compiled model data deserialization</title>
</head>
<body>
<h1>WebNN CWE-502: Untrusted Compiled Model Data Deserialization</h1>
<p>This page demonstrates the trust boundary where the GPU process deserializes
untrusted data from the (sandboxed) Compiler process without validation.</p>
<p>The actual vulnerability requires a compromised Compiler process to send
crafted bytes via the LoadCompiledGraph Mojo IPC. This page shows the normal
WebNN API path that routes through the vulnerable code; the exploitation itself
requires a compromised Compiler (out-of-web-reach).</p>
<pre id="log"></pre>
<script>
async function log(msg) {
  document.getElementById('log').textContent += msg + '\n';
}

async function demonstrate() {
  if (!('ml' in navigator)) {
    await log('WebNN (navigator.ml) is not available in this browser.');
    await log('Requires Chromium with --enable-features=WebMachineLearningNeuralNetwork on Windows.');
    return;
  }

  await log('navigator.ml available. Creating context...');

  try {
    // Create a WebNN context — on Windows with ORT backend, this goes through
    // WebNNContextProviderImpl → ContextImplOrt (or DispatchContextImplOrt
    // when the Compiler process is enabled).
    const context = await navigator.ml.createContext({
      deviceType: 'gpu',
      powerPreference: 'default',
    });
    await log('Context created.');

    // Build a minimal graph. When the Compiler process is enabled
    // (kWebNNCompilerProcess feature), the graph is compiled in the
    // sandboxed Compiler process, and the resulting compiled_model_data
    // buffer is sent back to the GPU process via the
    // WebNNModelLoader.LoadCompiledGraph Mojo IPC.
    //
    // At dispatch_context_impl_ort.cc:192, the buffer is passed directly
    // to GraphImplOrt::CreateSessionFromCompiledGraph (graph_impl_ort.cc:221),
    // which calls ort_api->CreateSessionFromArray with no format/structure
    // validation. ORT then calls model_proto.ParseFromArray
    // (inference_session.cc:1287) on the untrusted bytes.
    //
    // A COMPROMISED Compiler process could send arbitrary bytes here.
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {dataType: 'float32', shape: [1, 2]});
    const relu = builder.relu(input);
    const output = builder.output('output', {dataType: 'float32', shape: [1, 2]});

    await log('Building graph (triggers Compiler→GPU IPC with compiled_model_data)...');
    const graph = await builder.build({input: input}, {output: relu});
    await log('Graph built successfully.');

    // Run inference to exercise the full path.
    const inputBuffer = new Float32Array([−1.0, 2.0]);
    const outputBuffer = new Float32Array(2);
    const inputs = {input: inputBuffer};
    const outputs = {output: outputBuffer};
    await context.compute(graph, inputs, outputs);
    await log('Inference result: ' + Array.from(outputBuffer));
    await log('');
    await log('=== Vulnerability path exercised ===');
    await log('The compiled_model_data buffer crossed the Compiler→GPU trust');
    await log('boundary via LoadCompiledGraph Mojo IPC and was deserialized');
    await log('by ORT\'s protobuf parser (ParseFromArray) without validation.');
    await log('');
    await log('To exploit: a compromised Compiler process would need to send');
    await log('crafted bytes instead of valid ONNX protobuf data. This is not');
    await log('web-reachable — it requires compromising the Compiler process.');
  } catch (e) {
    await log('Error: ' + e.message);
    await log('');
    await log('Note: If the Compiler process is not enabled, graph building');
    await log('happens in-process via GraphImplOrt::CreateAndBuild, not through');
    await log('the vulnerable LoadCompiledGraph IPC path.');
  }
}

demonstrate();
</script>
</body>
</html>
```

## Test
_(not provided)_


## Suggested fix
Add validation before passing to ORT: (a) Check minimum size (e.g., 8 bytes for ONNX header) in dispatch_context_impl_ort.cc before calling CreateSessionFromCompiledGraph. (b) Validate ONNX protobuf magic bytes (0x08, 0x00 for ir_version field) at the start of the buffer. (c) Consider using a hardened protobuf parser with strict size limits and depth limits. (d) Add a fuzz-testing harness for the Compiler→GPU IPC path with malformed protobuf inputs to discover parser bugs. Example validation in graph_impl_ort.cc:221: `if (compiled_model_data.size() < 8) return error; const uint8_t* data = compiled_model_data.data(); if (data[0] != 0x08) return error; // Check ONNX magic`


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #637.
