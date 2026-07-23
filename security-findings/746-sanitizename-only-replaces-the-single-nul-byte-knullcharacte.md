# Security finding #746: SanitizeName only replaces the single NUL byte (kNullCharacter, lin…

**Summary:** SanitizeName only replaces the single NUL byte (kNullCharacter, lin…

**CWE IDs:** CWE-20: Improper Input Validation (residual control characters in ONNX names)
**Severity / Impact:** Names carrying raw control characters are serialized into the ONNX/OV IR name strings and re-parsed downstream by the OpenVINO EP. Depending on OV IR name normalization this can cause name mismatches between the operand→onnx-name maps (model_editor.cc:121,130) and the actual graph, or graph-construction faults, degrading to a service-process error/crash. Lower confidence than finding #1: the downstream OV EP name-parsing sensitivity could not be confirmed within the read budget.
**Affected location:** `targets/chromium/services/webnn/ort/graph_builder_ort.cc:165` — `SanitizeName()`
**Validated for repos:** chromiumWebnn
**Trust boundary:** web content (renderer) → Mojo IPC → WebNN service → ORT model editor → OV EP / OpenVINO IR

## Description / Root cause
SanitizeName only replaces the single NUL byte (kNullCharacter, line 163) and passes every other byte through verbatim. All other C0 control characters (U+0001–U+001F, U+007F), UTF-8 overlong/ill-formed sequences, and Unicode bidi overrides survive and are embedded directly into ONNX value-info / node names via name.c_str() in CreateOrtValueInfo (model_editor.cc:96-98), AddInitializerToGraph (model_editor.cc:202-204) and CreateNode (model_editor.cc:285-288). The comment at line 173-174 acknowledges only the NUL-start rejection path, showing other control chars were not considered.

**Validator analysis:** The vuln type CWE-20 is accurate: SanitizeName at graph_builder_ort.cc:165-170 constitutes incomplete input validation — it was written with only the NUL-start ORT restriction in mind (documented by the comment at line 173-174) and does not strip any other control characters. The data-flow path from web content is confirmed: renderer sets mojom::Operand::name (arbitrary UTF-8 string from JS), Mojo IPC delivers it to the service, GetOperandNameById() (line 396-400) calls GetOperandName() (line 172-177) which calls SanitizeName(), and the resulting string containing e.g. U+001F bytes flows into CreateValueInfo/AddInitializerToGraph/CreateNode as C-strings. The proposed fix (allowlist [A-Za-z0-9_.−] or strip all bytes < 0x20 and 0x7F before they reach model_editor) is correct and sufficient. However the stated impact — name mismatch / service crash downstream in OV EP — is speculative and unconfirmed; the flaw is a real input-validation gap in the Chromium WebNN service but the actual exploitability requires an ORT or OV EP parser that trips on residual control chars, which was not verified. The NUL case was already fixed; other control chars are a gap that should be closed defensively.

## Exploit / Proof of Concept
Create a WebNN operand whose name contains a byte such as U+001F (or a truncating-looking control char). SanitizeName leaves it intact; it is stored as the ONNX tensor name and copied into operand_input_name_to_onnx_input_name_map. Any downstream component that trims/normalizes control chars (ORT protobuf tooling or OV IR parser) then fails to resolve the name, producing a build/inference fault.

## Reproduction (steps)
```
Steps to reach the vulnerable code path from a web page:

1. Open a browser with WebNN ORT backend enabled (e.g. Chromium + --enable-features=WebMachineLearningNeuralNetwork with ORT backend active on Windows).
2. In a web page, create an MLContext and MLGraphBuilder:

```html
<!DOCTYPE html>
<html>
<body>
<script>
async function trigger() {
  const context = await navigator.ml.createContext({deviceType: 'cpu'});
  const builder = new MLGraphBuilder(context);
  // U+001F (unit separator) in operand name — passes through SanitizeName unchanged
  const controlCharName = 'input\x1fwith\x01control';
  const input = builder.input(controlCharName, {dataType: 'float32', shape: [1, 4]});
  const relu = builder.relu(input);
  // Build triggers GraphBuilderOrt::CreateAndBuild → GetOperandNameById →
  // GetOperandName → SanitizeName → model_editor CreateValueInfo with
  // 'input\x1fwith\x01control_<id>' as the ONNX tensor name
  try {
    const graph = await builder.build({'output': relu});
    console.log('Built graph (control chars survived into ONNX names)');
  } catch(e) {
    console.log('Error:', e);
  }
}
trigger();
</script>
</body>
</html>
```

3. The operand name 'input\x1fwith\x01control' flows via Mojo to the WebNN service. SanitizeName at graph_builder_ort.cc:165-170 only strips '\0'; the U+001F and U+0001 bytes survive. The resulting name string (e.g. 'input\x1fwith\x01control_0') is passed as the ONNX tensor name to ort_model_editor_api->CreateValueInfo (model_editor.cc:96-98) with verbatim control chars embedded.

Note: Whether this causes an observable crash or silent corruption depends on ORT/OV EP name handling which is unconfirmed. The finding is primarily the input-validation gap in SanitizeName.
```

## Test
_(not provided)_


## Suggested fix
Broaden SanitizeName to replace or reject the full set of disallowed characters (all bytes < 0x20, 0x7F, and validate UTF-8 well-formedness) rather than only `\0`, or restrict operand names to a conservative allowlist (e.g. [A-Za-z0-9_.-]) before they reach the ORT model editor.

## Models used

| Role | Provider / model |
| --- | --- |
| Idea | `claude-4-6-sonnet` |
| Tech lead | `claude-4-6-sonnet` |
| Researcher | `claude-4-8-opus` |
| Validator | `claude-4-6-sonnet` |


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #746.
