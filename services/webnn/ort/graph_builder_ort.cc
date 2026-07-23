Now I have the complete file. The fix is:
1. Remove the `kNullCharacter` constant (it becomes unused after the fix).
2. Rewrite `SanitizeName` to replace all C0 control characters (< 0x20) and DEL (0x7F) with `_`, instead of only replacing the NUL byte.

// Copyright 2025 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/ort/graph_builder_ort.h"

#include <array>
#include <numeric>
#include <ranges>

#include "base/notreached.h"
#include "base/strings/string_number_conversions.h"
#include "base/strings/string_util.h"
#include "base/types/expected_macros.h"
#include "base/types/fixed_array.h"
#include "services/webnn/ort/ort_data_type.h"
#include "services/webnn/public/cpp/graph_validation_utils.h"
#include "services/webnn/public/cpp/supported_data_types.h"
#include "services/webnn/public/mojom/webnn_error.mojom.h"
#include "services/webnn/public/mojom/webnn_graph.mojom.h"
#include "services/webnn/webnn_constant_operand.h"
#include "third_party/fp16/src/include/fp16.h"

namespace webnn::ort {

namespace {

// ArgMin/Max ops
constexpr base::cstring_view kOpTypeArgMin = "ArgMin";
constexpr base::cstring_view kOpTypeArgMax = "ArgMax";

// Element-wise binary ops
constexpr base::cstring_view kOpTypeAdd = "Add";
constexpr base::cstring_view kOpTypeSub = "Sub";
constexpr base::cstring_view kOpTypeMul = "Mul";
constexpr base::cstring_view kOpTypeDiv = "Div";
constexpr base::cstring_view kOpTypeMax = "Max";
constexpr base::cstring_view kOpTypeMin = "Min";
constexpr base::cstring_view kOpTypePow = "Pow";
constexpr base::cstring_view kOpTypeEqual = "Equal";
constexpr base::cstring_view kOpTypeGreater = "Greater";
constexpr base::cstring_view kOpTypeGreaterOrEqual = "GreaterOrEqual";
constexpr base::cstring_view kOpTypeLesser = "Less";
constexpr base::cstring_view kOpTypeLesserOrEqual = "LessOrEqual";
constexpr base::cstring_view kOpTypeLogicalAnd = "And";
constexpr base::cstring_view kOpTypeLogicalOr = "Or";
constexpr base::cstring_view kOpTypeLogicalXor = "Xor";

// Element-wise unary ops
constexpr base::cstring_view kOpTypeAbs = "Abs";
constexpr base::cstring_view kOpTypeCeil = "Ceil";
constexpr base::cstring_view kOpTypeCos = "Cos";
constexpr base::cstring_view kOpTypeExp = "Exp";
constexpr base::cstring_view kOpTypeFloor = "Floor";
constexpr base::cstring_view kOpTypeLog = "Log";
constexpr base::cstring_view kOpTypeIsNaN = "IsNaN";
constexpr base::cstring_view kOpTypeIsInfinite = "IsInf";
constexpr base::cstring_view kOpTypeLogicalNot = "Not";
constexpr base::cstring_view kOpTypeNeg = "Neg";
constexpr base::cstring_view kOpTypeRoundEven = "Round";
constexpr base::cstring_view kOpTypeSign = "Sign";
constexpr base::cstring_view kOpTypeSin = "Sin";
constexpr base::cstring_view kOpTypeTan = "Tan";
constexpr base::cstring_view kOpTypeIdentity = "Identity";
constexpr base::cstring_view kOpTypeSqrt = "Sqrt";
constexpr base::cstring_view kOpTypeErf = "Erf";
constexpr base::cstring_view kOpTypeReciprocal = "Reciprocal";
constexpr base::cstring_view kOpTypeCast = "Cast";

constexpr base::cstring_view kOpTypeBatchNormalization = "BatchNormalization";
constexpr base::cstring_view kOpTypeClamp = "Clip";
constexpr base::cstring_view kOpTypeConcat = "Concat";
constexpr base::cstring_view kOpTypeConv2d = "Conv";
constexpr base::cstring_view kOpTypeConvTranspose2d = "ConvTranspose";
constexpr base::cstring_view kOpTypeCumulativeSum = "CumSum";
constexpr base::cstring_view kOpTypeDequantizeLinear = "DequantizeLinear";
constexpr base::cstring_view kOpTypeElu = "Elu";
constexpr base::cstring_view kOpTypeExpand = "Expand";
constexpr base::cstring_view kOpTypeGather = "Gather";
constexpr base::cstring_view kOpTypeGatherElements = "GatherElements";
constexpr base::cstring_view kOpTypeGatherND = "GatherND";
constexpr base::cstring_view kOpTypeGelu = "Gelu";
constexpr base::cstring_view kOpTypeGemm = "Gemm";
constexpr base::cstring_view kOpTypeGru = "GRU";
constexpr base::cstring_view kOpTypeHardSigmoid = "HardSigmoid";
constexpr base::cstring_view kOpTypeHardSwish = "HardSwish";
constexpr base::cstring_view kOpTypeInstanceNormalization =
    "InstanceNormalization";
constexpr base::cstring_view kOpTypeLayerNormalization = "LayerNormalization";
constexpr base::cstring_view kOpTypeLeakyRelu = "LeakyRelu";
constexpr base::cstring_view kOpTypeLstm = "LSTM";
constexpr base::cstring_view kOpTypeMatMul = "MatMul";
constexpr base::cstring_view kOpTypePad = "Pad";
constexpr base::cstring_view kOpTypePRelu = "PRelu";
constexpr base::cstring_view kOpTypeQuantizeLinear = "QuantizeLinear";
constexpr base::cstring_view kOpTypeRelu = "Relu";
constexpr base::cstring_view kOpTypeResize = "Resize";
constexpr base::cstring_view kOpTypeReshape = "Reshape";
constexpr base::cstring_view kOpTypeScatterElements = "ScatterElements";
constexpr base::cstring_view kOpTypeScatterND = "ScatterND";
constexpr base::cstring_view kOpTypeSigmoid = "Sigmoid";
constexpr base::cstring_view kOpTypeSlice = "Slice";
constexpr base::cstring_view kOpTypeSoftmax = "Softmax";
constexpr base::cstring_view kOpTypeSoftplus = "Softplus";
constexpr base::cstring_view kOpTypeSoftsign = "Softsign";
constexpr base::cstring_view kOpTypeSplit = "Split";
constexpr base::cstring_view kOpTypeTanh = "Tanh";
constexpr base::cstring_view kOpTypeTile = "Tile";
constexpr base::cstring_view kOpTypeTranspose = "Transpose";
constexpr base::cstring_view kOpTypeTriangular = "Trilu";
constexpr base::cstring_view kOpTypeWhere = "Where";

// Pooling operations
constexpr base::cstring_view kOpTypeAveragePool2d = "AveragePool";
constexpr base::cstring_view kOpTypeMaxPool2d = "MaxPool";
constexpr base::cstring_view kOpTypeLpPool2d = "LpPool";

// Reduction operations
constexpr base::cstring_view kOpTypeReduceL1 = "ReduceL1";
constexpr base::cstring_view kOpTypeReduceL2 = "ReduceL2";
constexpr base::cstring_view kOpTypeReduceLogSum = "ReduceLogSum";
constexpr base::cstring_view kOpTypeReduceLogSumExp = "ReduceLogSumExp";
constexpr base::cstring_view kOpTypeReduceMax = "ReduceMax";
constexpr base::cstring_view kOpTypeReduceMean = "ReduceMean";
constexpr base::cstring_view kOpTypeReduceMin = "ReduceMin";
constexpr base::cstring_view kOpTypeReduceProd = "ReduceProd";
constexpr base::cstring_view kOpTypeReduceSum = "ReduceSum";
constexpr base::cstring_view kOpTypeReduceSumSquare = "ReduceSumSquare";

// Attributes
constexpr base::cstring_view kAttrActivations = "activations";
constexpr base::cstring_view kAttrAlpha = "alpha";
constexpr base::cstring_view kAttrAxis = "axis";
constexpr base::cstring_view kAttrBeta = "beta";
constexpr base::cstring_view kAttrBlockSize = "block_size";
constexpr base::cstring_view kAttrCeilMode = "ceil_mode";
constexpr base::cstring_view kAttrDilations = "dilations";
constexpr base::cstring_view kAttrDirection = "direction";
constexpr base::cstring_view kAttrEpsilon = "epsilon";
constexpr base::cstring_view kAttrExclusive = "exclusive";
constexpr base::cstring_view kAttrGroup = "group";
constexpr base::cstring_view kAttrHiddenSize = "hidden_size";
constexpr base::cstring_view kAttrKeepDims = "keepdims";
constexpr base::cstring_view kAttrKernelShape = "kernel_shape";
constexpr base::cstring_view kAttrLinearBeforeReset = "linear_before_reset";
constexpr base::cstring_view kAttrMode = "mode";
constexpr base::cstring_view kAttrNoopWithEmptyAxes = "noop_with_empty_axes";
constexpr base::cstring_view kAttrNumOutputs = "num_outputs";
constexpr base::cstring_view kAttrOutputPadding = "output_padding";
constexpr base::cstring_view kAttrP = "p";
constexpr base::cstring_view kAttrPads = "pads";
constexpr base::cstring_view kAttrPerm = "perm";
constexpr base::cstring_view kAttrReverse = "reverse";
constexpr base::cstring_view kAttrStrides = "strides";
constexpr base::cstring_view kAttrTo = "to";
constexpr base::cstring_view kAttrTransA = "transA";
constexpr base::cstring_view kAttrTransB = "transB";
constexpr base::cstring_view kAttrUpper = "upper";

constexpr base::cstring_view kInserted = "Inserted";
constexpr base::cstring_view kToEmulate = "ToEmulate";
constexpr base::cstring_view kUnderscore = "_";

std::string SanitizeName(std::string_view name) {
  std::string sanitized_name(name);
  for (char& c : sanitized_name) {
    // Replace C0 control characters (U+0000–U+001F) and DEL (U+007F) with
    // underscores to prevent them from being embedded in ORT value-info /
    // node names. The ORT CreateValueInfo API rejects names starting with a
    // null character, and other control characters can cause similar issues:
    // https://github.com/microsoft/onnxruntime/blob/7b5a93ef5f71ca58a1b6e4ae81b250e767756c68/onnxruntime/core/session/model_editor_c_api.cc#L29
    if (static_cast<unsigned char>(c) < 0x20 ||
        static_cast<unsigned char>(c) == 0x7F) {
      c = '_';
    }
  }
  return sanitized_name;
}

std::string GetOperandName(std::string_view name, OperandId id) {
  // ORT CreateValueInfo API rejects name starting with null character:
  // https://github.com/microsoft/onnxruntime/blob/7b5a93ef5f71ca58a1b6e4ae81b250e767756c68/onnxruntime/core/session/model_editor_c_api.cc#L29
  return base::JoinString(
      {SanitizeName(name), base::NumberToString(id.value())}, kUnderscore);
}

// Maps a DataType to a `ONNXTensorElementDataType`. Other `TensorTypeMap`
// overloads may be declared below as