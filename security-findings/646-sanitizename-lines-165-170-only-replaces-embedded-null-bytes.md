# Security finding #646: SanitizeName (lines 165-170) only replaces embedded null bytes (kNu…

**Summary:** SanitizeName (lines 165-170) only replaces embedded null bytes (kNu…

**CWE IDs:** CWE-20: Improper Input Validation
**Severity / Impact:** Untrusted control characters survive into ORT's internal std::string name fields (ModelEditorValueInfo::name, ModelEditorNode::node_name, input_names, output_names). ORT matches node input/output names against initializer and value-info names by std::string equality during graph resolution — a name containing embedded control characters could cause graph edge misidentification, cache key collisions in session/cache keyed by node names, or protobuf serialization issues when the model is serialized (ONNX protobuf string fields may have constraints on control characters). BIDI override characters could cause visual deception in diagnostic/logging output. A malicious web page using the WebNN API can trigger this by supplying operand names or operation labels containing path separators, newlines, or BIDI control characters.
**Affected location:** `targets/chromium/services/webnn/ort/graph_builder_ort.cc:165` — `SanitizeName()`
**Validated for repos:** chromiumWebnn, onnxruntime
**Trust boundary:** Renderer process → WebNN mojom IPC → browser process graph_builder_ort SanitizeName → ORT model editor C API

## Description / Root cause
SanitizeName (lines 165-170) only replaces embedded null bytes (kNullCharacter) with underscores via base::ReplaceChars. It does NOT filter any other control characters — path separators (/ \), newlines (\r \n \t), BIDI override codepoints (U+202E, U+202A-U+202D, U+200E/U+200F), or other C0/C1 control characters. The mojom types Operand.name (webnn_graph.mojom:56, a plain `string?`) and operation label fields (e.g., ArgMinMax.label at :81, plain `string`) have no charset constraints, so a malicious renderer can supply names containing arbitrary control characters. SanitizeName is the sole validation gate before these names reach ORT's C API: CreateValueInfo (model_editor_c_api.cc:28-55) only rejects nullptr/empty and copies `name` verbatim into `vi->name` (line 48); CreateNode (model_editor_c_api.cc:57-107) does zero validation on node_name (line 67: `n->node_name = node_name`). At model_editor.cc:286, node_name.c_str() faithfully reproduces the full string (since nulls were already removed) including all surviving control characters.

**Validator analysis:** The defect is real and reachable: SanitizeName (graph_builder_ort.cc:165-170) only replaces kNullCharacter via base::ReplaceChars, leaving all other control characters (CR, LF, TAB, path separators, BIDI overrides) intact. The mojom types Operand.name (webnn_graph.mojom:56, plain string?) and operation labels (e.g. ArgMinMax.label at :81, plain string) impose no charset constraints, so a malicious renderer can supply arbitrary control characters. These flow through GetOperandNameById (line 398) and GenerateNodeName (line 404) into model_editor_.AddNode/AddValueInfo calls, reaching the ORT C API CreateValueInfo (model_editor_c_api.cc:48: vi->name = name) and CreateNode (model_editor_c_api.cc:67: n->node_name = node_name), both of which copy the string verbatim with only null/empty checks (line 31). The vuln type (CWE-20) is accurate for the validation gap. However, the impact is overstated: no concrete evidence shows graph edge misidentification, cache key collisions, or path traversal — ORT uses std::string equality matching so embedded control chars would create unique names rather than collisions, and no code path was shown using node names for file path construction. The realistic impact is protobuf serialization issues (ONNX proto string fields with control chars) and log/diagnostic visual deception via BIDI overrides. The proposed allowlist fix (replace any char outside [A-Za-z0-9._-] with underscore) is correct and sufficient, and is the better approach over the minimal extend-ReplaceChars suggestion since an allowlist is more robust than a denylist.

## Exploit / Proof of Concept
A web page creates a WebNN graph via the MLGraphBuilder API, supplying an input operand with name containing control characters (e.g., "input\nmalicious" or "input/../../../etc/passwd"). The renderer sends this via mojom IPC to the browser process. GetOperandName (line 172-177) calls SanitizeName(name) which only strips null bytes, then appends the operand ID. The resulting name with embedded newlines/slashes is passed to CreateOrtValueInfo → CreateValueInfo (model_editor_c_api.cc:48: `vi->name = name`). Similarly, operation labels with control characters pass through GenerateNodeName → SanitizeName → CreateNode (model_editor_c_api.cc:67: `n->node_name = node_name`). The control characters are now in ORT's internal name tables used for graph resolution and protobuf serialization. For example, a label containing a path separator could cause confusion if the name is later used in file path construction for external data, and newlines could corrupt log output or protobuf serialization.

## Reproduction (html)
```
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>WebNN SanitizeName Control Character Repro</title></head>
<body>
<h1>WebNN SanitizeName Control Character Repro</h1>
<p>Open the console (DevTools) to observe results.</p>
<script>
(async () => {
  if (!navigator.ml) {
    document.body.innerHTML += '<p>navigator.ml not available. Enable WebNN via --enable-features=WebMachineLearningNeuralNetwork</p>';
    return;
  }
  try {
    const ctx = await navigator.ml.createContext();
    const builder = new MLGraphBuilder(ctx);

    // Input operand name containing embedded newline and path separator.
    // SanitizeName (graph_builder_ort.cc:165) only strips null bytes,
    // so these control characters survive into ORT's internal name fields.
    const desc = { type: 'float32', dimensions: [2, 2] };
    const input = builder.input('input\nmalicious', desc);
    const output = builder.output('output', desc);

    // An operation whose label also carries control characters —
    // GenerateNodeName (graph_builder_ort.cc:402) calls SanitizeName on it.
    const add = builder.add(input, input, { label: 'op\r\n/..\\..\\evil' });
    builder.reshape(add, [2, 2], { label: 'reshape\tinject' });

    try {
      const graph = await builder.build({ output });
      document.body.innerHTML += '<p style="color:orange">Graph built successfully — control characters passed SanitizeName into ORT internal name fields without rejection.</p>';
      console.log('Graph built. Control characters survived SanitizeName validation gate.');
    } catch (e) {
      document.body.innerHTML += `<p style="color:red">Build failed: ${e.message}</p>`;
      console.error('Build error:', e);
    }
  } catch (e) {
    document.body.innerHTML += `<p style="color:red">Error: ${e.message}</p>`;
    console.error(e);
  }
})();
</script>
</body>
</html>
```

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-20 in graph_builder_ort.cc:165-170 (SanitizeName).
// The flaw: SanitizeName only replaces null bytes, allowing control characters
// (CR, LF, TAB, path separators, BIDI overrides) to survive into ORT C API
// CreateValueInfo (model_editor_c_api.cc:48) and CreateNode (:67).
//
// This test verifies that after the fix (allowlist-based sanitization),
// operand names and labels containing control characters are normalized to
// safe characters and do NOT appear verbatim in ORT internal name fields.
//
// Target: chromium unit tests for services/webnn/ort
// Build: out/Release/webnn_ort_unittests
// Run: out/Release/webnn_ort_unittests --gtest_filter=GraphBuilderOrtTest.SanitizeNameRejectsControlChars
//
// Note: This is a skeleton because the exact test harness header names and
// fixture class names for graph_builder_ort unit tests were not read; adapt
// to the actual test file under services/webnn/ort/tests/.

#include "testing/gtest/include/gtest/gtest.h"
#include "services/webnn/ort/graph_builder_ort.h"  // TODO: verify exact include path

// TODO: If SanitizeName is not publicly accessible, this test needs to go
// through a higher-level API (e.g. GraphBuilderOrt::CreateAndBuild with a
// mojom::GraphInfo containing control-char names) and assert that the
// resulting model does not contain raw control characters in any name field.
//
// A unit test approach:
// 1. Construct a mojom::GraphInfo with an Operand whose name contains
//    "input\nmalicious" and an operation whose label contains "op\r/..\\evil".
// 2. Call GraphBuilderOrt::CreateAndBuild(...).
// 3. After the fix: expect success and verify that the produced ORT model's
//    value_info and node names do NOT contain control characters.
//    Before the fix: the names would contain raw control characters,
//    which could cause issues in protobuf serialization or logging.

TEST(GraphBuilderOrtTest, SanitizeNameRejectsControlChars) {
  // TODO: Construct a minimal mojom::GraphInfo with:
  //   - An input operand named "input\nmalicious"
  //   - An ArgMinMax op with label "op\r\n/..\\..\\evil"
  // Then call GraphBuilderOrt::CreateAndBuild and inspect the resulting
  // ModelInfo's name fields.
  //
  // After the fix (allowlist [A-Za-z0-9._-]):
  //   EXPECT_EQ(result_name.find('\n'), std::string::npos);
  //   EXPECT_EQ(result_name.find('/'), std::string::npos);
  //   EXPECT_EQ(result_name.find('\\'), std::string::npos);
  //
  // Before the fix (only null stripped), these assertions would FAIL
  // because the control characters survive.
  GTEST_SKIP() << "Skeleton: requires mojom::GraphInfo fixture construction "
                  "and access to ORT model editor internals to inspect names. "
                  "See comments above for the test logic.";
}
```
**Build / run:** Build target: webnn_ort_unittests (or the appropriate unit test binary for services/webnn/ort). Run: out/Release/webnn_ort_unittests --gtest_filter=GraphBuilderOrtTest.SanitizeNameRejectsControlChars. Expected: pre-fix, the test would fail because control characters survive SanitizeName into ORT name fields; post-fix (allowlist sanitization), all control characters are replaced with underscores and the test passes.

## Suggested fix
Replace SanitizeName with a function that validates the entire character set, not just null bytes. For example: reject or replace all C0/C1 control characters (0x00-0x1F, 0x7F-0x9F), path separators (/ and \), and BIDI override codepoints (U+202A-U+202E, U+200E, U+200F). A minimal fix would be to extend the ReplaceChars call to include all control characters: `base::ReplaceChars(sanitized_name, {kNullCharacter, "\r", "\n", "\t", "/", "\\", ...}, kUnderscore, &sanitized_name);`. Better yet, use an allowlist approach: replace any character outside [A-Za-z0-9._-] with underscore, which is the safe character set for ONNX names.

## Models used

| Role | Provider / model |
| --- | --- |
| Idea | `claude-4-6-sonnet` |
| Tech lead | `openrouter,@preset/glm-5-2-no-reasoning` |
| Researcher | `openrouter,@preset/glm-5-2-no-reasoning` |
| Validator | `openrouter,@preset/glm-5-2-no-reasoning` |


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #646.
