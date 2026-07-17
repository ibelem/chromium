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
