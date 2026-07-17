# Security finding #635: `LoadCompiledGraph` trusts `compiled_graph->inputs.size()` and `com…

**Summary:** `LoadCompiledGraph` trusts `compiled_graph->inputs.size()` and `com…

**CWE IDs:** CWE-20: Improper Input Validation
**Severity / Impact:** Any Compiler process able to invoke this IPC can force the GPU process to pre-allocate vectors sized to an attacker-chosen value, causing excessive memory consumption (CWE-400/CWE-789) in the more-privileged GPU process. Combined with the absent mojom-level `MaxSize`, this converts a Compiler-process compromise into a GPU-process DoS.
**Affected location:** `targets/chromium/services/webnn/ort/dispatch_context_impl_ort.cc:162` — `DispatchContextImplOrt::LoadCompiledGraph()`
**Validated for repos:** chromiumWebnn
**Trust boundary:** Mojo IPC from sandboxed Compiler process to GPU process via WebNNModelLoader.LoadCompiledGraph

## Description / Root cause
`LoadCompiledGraph` trusts `compiled_graph->inputs.size()` and `compiled_graph->outputs.size()` directly, using them in `reserve()` calls (lines 162-163, 176-177) and to construct `base::flat_map`s (lines 168-171, 182-185) with no validation that the size is within a reasonable bound for a WebNN graph. There is no upper-bound check before the allocation-amplifying `reserve()` calls.

**Validator analysis:** The flaw is real and reachable from the stated trust boundary. dispatch_context_impl_ort.cc:162-185 trusts compiled_graph->inputs.size() and compiled_graph->outputs.size() from the Mojo IPC payload (webnn_model_loader.mojom:30-31 has no [MaxSize]) and passes them directly to four reserve() calls. In Chromium, PartitionAlloc CHECK-fails on oversized allocations, crashing the GPU process. The vulnType 'CWE-20: Improper Input Validation' is accurate; the impact (CWE-400/789, DoS of the more-privileged GPU process from a compromised Compiler) is correctly described. The 'allocation amplification' framing is slightly overstated — reserve() allocates the same memory that push_back growth would reach — but the core defect (no size cap on untrusted IPC data before allocation) is genuine. The proposed fix (early-exit with an error if sizes exceed a constant like kMaxCompiledGraphPortCount) is correct and sufficient; it should be placed before line 162 and also enforced at the mojom level via [MaxSize] annotations.

## Exploit / Proof of Concept
A compromised Compiler invokes `WebNNModelLoader.LoadCompiledGraph` with a `CompiledGraph` whose `inputs` map has `size()` set to a very large value (e.g., 2^28). The four `reserve()` calls at dispatch_context_impl_ort.cc:162-163 and 176-177 each allocate ~multi-GB backing storage before any element is copied, exhausting GPU-process memory and crashing the process.

## Reproduction (steps)
```
Reproduction steps (requires privileged access to the Compiler IPC, not reproducible from web content alone):

1. Set up a Chromium build with --enable-features=WebMachineLearningNeuralNetwork.
2. Identify the Compiler process IPC endpoint for WebNNModelLoader (this is the Mojo interface exposed by the GPU process to the Compiler process, defined in webnn_model_loader.mojom:48-54).
3. Craft a Mojo message for LoadCompiledGraph() with a CompiledGraph struct where the `inputs` map claims 2^26+ entries (each with a minimal CompiledOperandDescriptor).
4. Send this message to the GPU process's WebNNModelLoader receiver.
5. Observe GPU process crash in PartitionAlloc or std::vector::reserve() at dispatch_context_impl_ort.cc:162-163 due to oversized allocation.

Why a pure web HTML trigger is not practical: A web page can only submit graphs via MLGraphBuilder.input()/output(), which the honest Compiler then compiles. The sizes in the CompiledGraph legitimately match the web graph's I/O count, and creating millions of builder.input() calls would exhaust renderer resources first. The threat model specifically requires a compromised Compiler process (e.g., via a separate exploit in the ORT/ML runtime running in the Compiler sandbox).
```

## Test
_(not provided)_


## Suggested fix
Before the `reserve()` calls, validate `compiled_graph->inputs.size()` and `compiled_graph->outputs.size()` against a reasonable constant (e.g., `kMaxCompiledGraphPortCount`, matching WebNN spec limits on graph I/O ports). If exceeded, return `mojom::Error::New(mojom::Error::Code::kInvalidArgument, "too many input/output ports")` early, before any allocation proportional to the untrusted size.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #635.
