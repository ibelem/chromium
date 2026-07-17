# Security finding #634: The `binding_name` field of `CompiledOperandDescriptor` (line 17) a…

**Summary:** The `binding_name` field of `CompiledOperandDescriptor` (line 17) a…

**CWE IDs:** CWE-20: Improper Input Validation
**Severity / Impact:** A compromised Compiler process can send extremely long binding names (e.g., hundreds of MB each) which will be allocated in the GPU process. Repeated across many map entries, this amplifies per-message memory consumption far beyond the serialized message size (each string is separately heap-allocated). Downstream ORT calls may also misbehave on unexpectedly long C-string arguments.
**Affected location:** `targets/chromium/services/webnn/public/mojom/webnn_model_loader.mojom:17`
**Validated for repos:** chromiumWebnn
**Trust boundary:** Mojo IPC from sandboxed Compiler process to GPU process via WebNNModelLoader.LoadCompiledGraph

## Description / Root cause
The `binding_name` field of `CompiledOperandDescriptor` (line 17) and the map keys in `CompiledGraph` (line 30-31) are declared as unbounded `string` with no `[MaxSize]` attribute. No length check is performed in `DispatchContextImplOrt::LoadCompiledGraph` (dispatch_context_impl_ort.cc:164-166, 178-180) before these strings are `std::move`d into `std::pair<std::string, std::string>` vectors and later passed as C strings to ORT API calls.

**Validator analysis:** The core claim is accurate: webnn_model_loader.mojom:17 declares `string binding_name` with no [MaxSize] attribute, and the map keys at lines 30-31 are also unbounded. dispatch_context_impl_ort.cc:164-166 (inputs) and 178-180 (outputs) iterate these maps and move the strings directly into std::vector<std::pair<std::string,std::string>> without any length validation, then pass them to GraphImplOrt::CreateSessionFromCompiledGraph which stores them in flat_maps (graph_impl_ort.cc:215-232), and they are later used as .c_str() pointers at graph_impl_ort.cc:88,99 for ORT API calls. The trust boundary is correct: this is Mojo IPC from a sandboxed Compiler process to the GPU process. However, the stated impact is overstated. Mojo messages have a hard maximum size (typically 64MB per message), so a single message cannot contain 'hundreds of MB each' strings across hundreds of entries — the amplification ratio is roughly 1:1 (message size ≈ memory consumed during deserialization). The move semantics at lines 165 and 179 avoid extra copies, not amplify them. The real concern is defense-in-depth: the lack of [MaxSize] constraints means a compromised Compiler can consume a significant fraction of the GPU process's memory budget in a single IPC message, and downstream ORT calls may encounter unexpectedly long C-string operands. The proposed fix — adding [MaxSize=N] to the mojom fields and adding a matching length check in LoadCompiledGraph — is correct and sufficient. A recommended value would be 256 bytes (as proposed), which is generous for ML operand names but prevents the DoS scenario.

## Exploit / Proof of Concept
A compromised Compiler sends a `CompiledGraph` whose `inputs` map contains a few hundred entries whose keys and `binding_name` fields are each ~10 MB strings. The GPU process allocates copies of each during deserialization and again when they are moved into vectors at dispatch_context_impl_ort.cc:165 and 179, leading to excessive GPU-process memory consumption and potential OOM termination.

## Reproduction (steps)
```
Reproduction of this flaw from a WebNN web page is NOT directly possible because the attack requires a COMPROMISED Compiler process to emit crafted Mojo IPC messages. A normal WebNN page calls navigator.ml.createContext() and context.build(), which triggers legitimate model compilation that produces well-formed, reasonably-sized binding names. The Compiler process is a sandboxed utility that a web page cannot control or compromise from script.

To verify the fix manually:
1. Build Chromium with ASan: gn gen out/Asan --args='is_asan=true'
2. In services/webnn/public/mojom/webnn_model_loader.mojom, confirm that [MaxSize=256] has been added to binding_name and map keys.
3. In dispatch_context_impl_ort.cc, confirm that LoadCompiledGraph rejects entries where name.size() > 256 or binding_name.size() > 256, returning mojom::Error early.
4. Write a Mojo fuzzer or test that constructs a mojom::CompiledGraph with binding_name strings of 1000+ bytes and 50+ map entries, then invoke WebNNModelLoader.LoadCompiledGraph and verify it returns an error (or verify the Mojo validator rejects the message due to [MaxSize] violation) rather than allocating the strings in the GPU process.
```

## Test
_(not provided)_


## Suggested fix
Add `[MaxSize=kMaxBindingNameLength]` to the `binding_name` field (e.g., `string binding_name [(MaxSize=256)]`) and to the map-key string in webnn_model_loader.mojom. Also add a belt-and-braces length check in `DispatchContextImplOrt::LoadCompiledGraph` that rejects entries with binding names or map keys exceeding a reasonable bound, returning a `mojom::Error` early.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #634.
