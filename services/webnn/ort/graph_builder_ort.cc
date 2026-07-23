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
constexpr std::string_view kNullCharacter("\0", 1);

// Maximum permitted byte length for an operand name. Names longer than this
// are rejected at the validator (webnn_graph_builder_impl.cc) before reaching
// here; the DCHECK below is a defence-in-depth guard.
constexpr size_t kMaxOperandNameLength = 4096;

std::string SanitizeName(std::string_view name) {
  DCHECK_LE(name.size(), kMaxOperandNameLength);
  std::string sanitized_name(name);
  base::ReplaceChars(sanitized_name, kNullCharacter, kUnderscore,
                     &sanitized_name);
  return sanitized_name;
}

std::string GetOperandName(std::string_view name, OperandId id) {
  // ORT CreateValueInfo API rejects name starting with null character:
  // https://github.com/microsoft/onnxruntime/blob/7b5a93ef5f71ca58a1b6e4ae81b250e767756c68/onnxruntime/core/session/model_editor_c_api.cc#L29
  return base::JoinString(
      {SanitizeName(name), base::NumberToString(id.value())}, kUnderscore);
}

// Maps a DataType to a `ONNXTensorElementDataType`. Other `TensorTypeMap`
// overloads may be declared below as needed.
//
// Example: TensorTypeMap<uint32_t>::value ->
// ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
template <typename DataType>
  requires internal::IsSupportedTensorType<DataType>
struct TensorTypeMap;

template <>
struct TensorTypeMap<float> {
  static constexpr ONNXTensorElementDataType value =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};

// Use uint16_t to carry bits of float16.
template <>
struct TensorTypeMap<uint16_t> {
  static constexpr ONNXTensorElementDataType value =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
};

template <>
struct TensorTypeMap<int32_t> {
  static constexpr ONNXTensorElementDataType value =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
};

template <>
struct TensorTypeMap<uint32_t> {
  static constexpr ONNXTensorElementDataType value =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
};

template <>
struct TensorTypeMap<int64_t> {
  static constexpr ONNXTensorElementDataType value =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
};

template <>
struct TensorTypeMap<uint64_t> {
  static constexpr ONNXTensorElementDataType value =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
};

template <>
struct TensorTypeMap<int8_t> {
  static constexpr ONNXTensorElementDataType value =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
};

template <>
struct TensorTypeMap<uint8_t> {
  static constexpr ONNXTensorElementDataType value =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
};

// Calculate the output_padding according to the ONNX ConvTranspose2d
// documentation:
// https://onnx.ai/onnx/operators/onnx__ConvTranspose.html#summary
int64_t CalculateOutputPaddingSize(int64_t input_size,
                                   int64_t filter_size,
                                   int64_t stride,
                                   int64_t dilation,
                                   int64_t pad_begin,
                                   int64_t pad_end,
                                   int64_t output_size) {
  const auto output_padding =
      base::CheckedNumeric(output_size) - stride * (input_size - 1) -
      ((filter_size - 1) * dilation + 1) + pad_begin + pad_end;
  // `output_padding` is validated by
  // `ValidateAndCalculateConvTranspose2dOutputSizes()`. Because Conv2d mojo
  // struct doesn't include `output_padding`, for ORT backend, we need to
  // re-compute it by using other attributes.
  CHECK(output_padding.IsValid());
  return output_padding.ValueOrDie();
}

void CheckReduceInputSupported(const DataTypeLimits& data_type_limits,
                               mojom::Reduce::Kind kind,
                               const OperandDescriptor& input_descriptor) {
  switch (kind) {
    case mojom::Reduce::Kind::kL1:
      CHECK(data_type_limits.reduce_l1_input.Supports(input_descriptor));
      break;
    case mojom::Reduce::Kind::kL2:
      CHECK(data_type_limits.reduce_l2_input.Supports(input_descriptor));
      break;
    case mojom::Reduce::Kind::kLogSum:
      CHECK(data_type_limits.reduce_log_sum_input.Supports(input_descriptor));
      break;
    case mojom::Reduce::Kind::kLogSumExp:
      CHECK(
          data_type_limits.reduce_log_sum_exp_input.Supports(input_descriptor));
      break;
    case mojom::Reduce::Kind::kMax:
      CHECK(data_type_limits.reduce_max_input.Supports(input_descriptor));
      break;
    case mojom::Reduce::Kind::kMean:
      CHECK(data_type_limits.reduce_mean_input.Supports(input_descriptor));
      break;
    case mojom::Reduce::Kind::kMin:
      CHECK(data_type_limits.reduce_min_input.Supports(input_descriptor));
      break;
    case mojom::Reduce::Kind::kProduct:
      CHECK(data_type_limits.reduce_product_input.Supports(input_descriptor));
      break;
    case mojom::Reduce::Kind::kSum:
      CHECK(data_type_limits.reduce_sum_input.Supports(input_descriptor));
      break;
    case mojom::Reduce::Kind::kSumSquare:
      CHECK(
          data_type_limits.reduce_sum_square_input.Supports(input_descriptor));
      break;
  }
}

base::cstring_view MapReduceKindToOrtOpType(mojom::Reduce::Kind kind) {
  switch (kind) {
    case mojom::Reduce::Kind::kL1:
      return kOpTypeReduceL1;
    case mojom::Reduce::Kind::kL2:
      return kOpTypeReduceL2;
    case mojom::Reduce::Kind::kLogSum:
      return kOpTypeReduceLogSum;
    case mojom::Reduce::Kind::kLogSumExp:
      return kOpTypeReduceLogSumExp;
    case mojom::Reduce::Kind::kMax:
      return kOpTypeReduceMax;
    case mojom::Reduce::Kind::kMean:
      return kOpTypeReduceMean;
    case mojom::Reduce::Kind::kMin:
      return kOpTypeReduceMin;
    case mojom::Reduce::Kind::kProduct:
      return kOpTypeReduceProd;
    case mojom::Reduce::Kind::kSum:
      return kOpTypeReduceSum;
    case mojom::Reduce::Kind::kSumSquare:
      return kOpTypeReduceSumSquare;
  }
}

const std::vector<base::cstring_view> GetRecurrentNetworkActivations(
    std::vector<mojom::RecurrentNetworkActivation> activations,
    bool is_bidirectional) {
  std::vector<base::cstring_view> activation_list;
  for (const auto& activation : activations) {
    switch (activation) {
      case mojom::RecurrentNetworkActivation::kRelu:
        activation_list.push_back("Relu");
        break;
      case mojom::RecurrentNetworkActivation::kSigmoid:
        activation_list.push_back("Sigmoid");
        break;
      case mojom::RecurrentNetworkActivation::kTanh:
        activation_list.push_back("Tanh");
        break;
      default:
        NOTREACHED() << "Unsupported recurrent network activation function.";
    }
  }
  if (is_bidirectional) {
    activation_list.insert(activation_list.end(), activation_list.begin(),
                           activation_list.end());
  }
  return activation_list;
}

const base::cstring_view GetRecurrentNetworkDirection(
    mojom::RecurrentNetworkDirection direction) {
  switch (direction) {
    case mojom::RecurrentNetworkDirection::kForward:
      return "forward";
    case mojom::RecurrentNetworkDirection::kBackward:
      return "reverse";
    case mojom::RecurrentNetworkDirection::kBoth:
      return "bidirectional";
    default:
      NOTREACHED() << "Unsupported recurrent network activation direction.";
  }
}

}  // namespace

// static
base::expected<std::unique_ptr<ModelEditor::ModelInfo>, mojom::ErrorPtr>
GraphBuilderOrt::CreateAndBuild(
    const mojom::GraphInfo& graph_info,
    ContextProperties context_properties,
    base::flat_map<OperandId, std::unique_ptr<WebNNConstantOperand>>
        constant_operands,
    std::optional<uint32_t> batched_matmul_k_dimension_limit) {
  GraphBuilderOrt graph_builder(graph_info, std::move(context_properties),
                                std::move(constant_operands),
                                std::move(batched_matmul_k_dimension_limit));
  return graph_builder.BuildModel();
}

GraphBuilderOrt::GraphBuilderOrt(
    const mojom::GraphInfo& graph_info,
    ContextProperties context_properties,
    base::flat_map<OperandId, std::unique_ptr<WebNNConstantOperand>>
        constant_operands,
    std::optional<uint32_t> batched_matmul_k_dimension_limit)
    : graph_info_(graph_info),
      constant_operands_(std::move(constant_operands)),
      context_properties_(std::move(context_properties)),
      batched_matmul_k_dimension_limit_(
          std::move(batched_matmul_k_dimension_limit)) {}

GraphBuilderOrt::~GraphBuilderOrt() = default;

const mojom::Operand& GraphBuilderOrt::GetOperand(OperandId operand_id) const {
  return *graph_info_->operands.at(operand_id.value());
}

std::string GraphBuilderOrt::GetOperandNameById(OperandId operand_id) const {
  const mojom::Operand& operand = GetOperand(operand_id);
  return GetOperandName(operand.name.has_value() ? *operand.name : "",
                        operand_id);
}

std::string GraphBuilderOrt::GenerateNodeName(std::string_view label) {
  return base::JoinString(
      {SanitizeName(label), base::NumberToString(next_operation_id_++)},
      kUnderscore);
}

std::string GraphBuilderOrt::GenerateEmulatedOpLabel(
    base::cstring_view op_type,
    std::string_view original_label,
    std::string_view additional_tag) {
  return base::JoinString({kInserted, op_type, additional_tag, kToEmulate,
                           SanitizeName(original_label)},
                          kUnderscore);
}

std::string GraphBuilderOrt::GenerateOperandName() {
  next_operand_id_++;
  CHECK(next_operand_id_.IsValid());
  return base::JoinString(
      {kInserted, base::NumberToString(
                      static_cast<uint32_t>(next_operand_id_.ValueOrDie()))},
      kUnderscore);
}

template <typename DataType>
  requires internal::IsSupportedTensorType<DataType>
std::string GraphBuilderOrt::CreateInitializer(
    base::span<const int64_t> shape,
    base::span<const DataType> data) {
  std::string name = GenerateOperandName();
  base::span<const uint8_t> byte_span;
  if constexpr (std::floating_point<DataType>) {
    // Floating point types do not have unique object representations, but
    // this code appears to be using a byte span to type-erase, which is fine.
    byte_span = base::as_byte_span(base::allow_nonunique_obj, data);
  } else {
    byte_span = base::as_byte_span(data);
  }

  model_editor_.AddInitializer(name, TensorTypeMap<DataType>::value, shape,
                               byte_span);
  return name;
}

template <typename DataType>
  requires internal::IsSupportedTensorType<DataType>
std::string GraphBuilderOrt::CreateScalarInitializer(const DataType& value) {
  return CreateInitializer<DataType>(
      /*shape=*/{}, base::span_from_ref(value));
}

template <typename DataType>
  requires internal::IsSupportedTensorType<DataType>
std::string GraphBuilderOrt::Create1DInitializer(
    base::span<const DataType> data) {
  std::array<int64_t, 1> shape = {base::checked_cast<int64_t>(data.size())};
  return CreateInitializer<DataType>(shape, data);
}

std::string GraphBuilderOrt::CreateInt64InitializerForUint32Array(
    base::span<const uint32_t> array) {
  std::array<int64_t, 1> array_dims = {
      base::checked_cast<int64_t>(array.size())};
  base::FixedArray<int64_t> array_value(array.begin(), array.end());
  return CreateInitializer<int64_t>(array_dims, array_value);
}

std::string GraphBuilderOrt::CreateInitializerForFloat(
    OperandDataType data_type,
    base::span<const uint32_t> shape,
    float value) {
  base::CheckedNumeric<size_t> checked_operand_size =
      std::accumulate(shape.begin(), shape.end(),
                      base::CheckedNumeric<size_t>(1), std::multiplies());
  size_t operand_size = checked_operand_size.ValueOrDie();
  base::FixedArray<int64_t> int64_shape(shape.begin(), shape.end());
  switch (data_type) {
    case OperandDataType::kFloat32: {
      base::FixedArray<float> data(operand_size, value);
      return CreateInitializer<float>(int64_shape, data);
    }
    case OperandDataType::kFloat16: {
      base::FixedArray<uint16_t> data(operand_size,
                                      fp16_ieee_from_fp32_value(value));
      return CreateInitializer<uint16_t>(int64_shape, data);
    }
    case OperandDataType::kInt32: {
      base::FixedArray<int32_t> data(operand_size,
                                     base::saturated_cast<int32_t>(value));
      return CreateInitializer<int32_t>(int64_shape, data);
    }
    case OperandDataType::kUint32: {
      base::FixedArray<uint32_t> data(operand_size,
                                      base::saturated_cast<uint32_t>(value));
      return CreateInitializer<uint32_t>(int64_shape, data);
    }
    case OperandDataType::kInt64: {
      base::FixedArray<int64_t> data(operand_size,
                                     base::saturated_cast<int64_t>(value));
      return CreateInitializer<int64_t>(int64_shape, data);
    }
    case OperandDataType::kUint64: {
      base::FixedArray<uint64_t> data(operand_size,
                                      base::saturated_cast<uint64_t>(value));
      return CreateInitializer<uint64_t>(int64_shape, data);
    }
    case OperandDataType::kInt8: {
      base::FixedArray<int8_t> data(operand_size,
                                    base::saturated_cast<int8_t>(value));
      return CreateInitializer<int8_t>(int64_shape, data);
    }
    case OperandDataType::kUint8: {
      base::FixedArray<uint8_t> data(operand_size,
                                     base::saturated_cast<uint8_t>(value));
      return CreateInitializer<uint8_t>(int64_shape, data);
    }
    case OperandDataType::kInt4:
    case OperandDataType::kUint4: {
      NOTREACHED();
    }
  }
}

std::string GraphBuilderOrt::CreateScalarInitializer(OperandDataType data_type,
                                                     const MLNumber& value) {
  switch (data_type) {
    case OperandDataType::kFloat32:
      return CreateScalarInitializer(value.AsFloat32());
    case OperandDataType::kFloat16:
      return CreateScalarInitializer(value.AsFloat16());
    case OperandDataType::kInt32:
      return CreateScalarInitializer(value.AsInt32());
    case OperandDataType::kUint32:
      return CreateScalarInitializer(value.AsUint32());
    case OperandDataType::kInt64:
      return CreateScalarInitializer(value.AsInt64());
    case OperandDataType::kUint64:
      return CreateScalarInitializer(value.AsUint64());
    case OperandDataType::kInt8:
      return CreateScalarInitializer(value.AsInt8());
    case OperandDataType::kUint8:
      return CreateScalarInitializer(value.AsUint8());
    case OperandDataType::kInt4:
    case OperandDataType::kUint4: {
      NOTREACHED();
    }
  }
}

std::string GraphBuilderOrt::CreateOneInitializer(
    OperandDataType data_type,
    base::span<const uint32_t> shape) {
  return CreateInitializerForFloat(data_type, shape, 1.0f);
}

std::string GraphBuilderOrt::CreateZeroInitializer(
    OperandDataType data_type,
    base::span<const uint32_t> shape) {
  return CreateInitializerForFloat(data_type, shape, 0.0f);
}

std::string GraphBuilderOrt::TransposeRnnWeightOrBiasLayout(
    base::cstring_view weight_or_bias,
    base::span<const uint32_t> permutation) {
  size_t num_gates = permutation.size();

  // Use Split operator to split the weight/bias into num_gates slices.
  std::vector<std::string> gate_names;
  gate_names.reserve(num_gates);
  for (size_t i = 0; i < num_gates; i++) {
    gate_names.push_back(GenerateOperandName());
  }
  constexpr int64_t axis = 1;
  std::array<ScopedOrtOpAttr, 2> split_attrs = {
      model_editor_.CreateAttribute(kAttrAxis, axis),
      model_editor_.CreateAttribute(kAttrNumOutputs,
                                    static_cast<int64_t>(num_gates))};
  std::array<const char*, 1> split_inputs = {weight_or_bias.c_str()};
  std::vector<const char*> split_outputs;
  split_outputs.reserve(num_gates);
  for (const auto& gate_name : gate_names) {
    split_outputs.push_back(gate_name.c_str());
  }
  std::string split_node_name = GenerateNodeName(
      base::JoinString({kInserted, kOpTypeSplit}, kUnderscore));
  model_editor_.AddNode(kOpTypeSplit, split_node_name, split_inputs,
                        split_outputs, split_attrs);

  // Use Concat operator to concatenate the slices in the order of permutation.
  std::vector<const char*> concat_inputs;
  concat_inputs.reserve(num_gates);
  for (uint32_t index : permutation) {
    concat_inputs.push_back(gate_names[index].c_str());
  }
  std::string concat_output = GenerateOperandName();
  std::array<const char*, 1> concat_outputs = {concat_output.c_str()};
  std::array<ScopedOrtOpAttr, 1> concat_attrs = {
      model_editor_.CreateAttribute(kAttrAxis, axis)};
  std::string concat_node_name = GenerateNodeName(
      base::JoinString({kInserted, kOpTypeConcat}, kUnderscore));
  model_editor_.AddNode(kOpTypeConcat, concat_node_name, concat_inputs,
                        concat_outputs, concat_attrs);

  return concat_output;
}

void GraphBuilderOrt::AddCastNode(base::cstring_view node_name,
                                  base::cstring_view input,
                                  base::cstring_view output,
                                  ONNXTensorElementDataType to_data_type) {
  std::array<const char*, 1> inputs = {input.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};
  int64_t attr_to_data = static_cast<int64_t>(to_data_type);
  std::array<ScopedOrtOpAttr, 1> attributes = {
      model_editor_.CreateAttribute(kAttrTo, attr_to_data)};

  model_editor_.AddNode(kOpTypeCast, node_name, inputs, outputs, attributes);
}

std::string GraphBuilderOrt::CreateCastNode(
    base::cstring_view input,
    ONNXTensorElementDataType to_data_type) {
  const std::string output = GenerateOperandName();
  InsertCastNode(input, output, to_data_type);
  return output;
}

void GraphBuilderOrt::InsertCastNode(base::cstring_view input,
                                     base::cstring_view output,
                                     ONNXTensorElementDataType to_data_type) {
  const std::string node_name =
      GenerateNodeName(base::JoinString({kInserted, kOpTypeCast}, kUnderscore));
  AddCastNode(node_name, input, output, to_data_type);
}

void GraphBuilderOrt::AddExpandNode(base::cstring_view node_name,
                                    base::cstring_view input,
                                    base::cstring_view output,
                                    base::span<const uint32_t> shape) {
  // `new_shape` should be the name of an int64 tensor that specifies the
  // output's shape.
  const std::string new_shape = CreateInt64InitializerForUint32Array(shape);

  std::array<const char*, 2> inputs = {input.c_str(), new_shape.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  model_editor_.AddNode(kOpTypeExpand, node_name, inputs, outputs);
}

std::string GraphBuilderOrt::CreateExpandNode(
    base::cstring_view input,
    base::span<const uint32_t> shape) {
  const std::string node_name = GenerateNodeName(
      base::JoinString({kInserted, kOpTypeExpand}, kUnderscore));
  const std::string output = GenerateOperandName();

  AddExpandNode(node_name, input, output, shape);
  return output;
}

void GraphBuilderOrt::AddResizeNode(base::cstring_view node_name,
                                    base::cstring_view input,
                                    base::cstring_view scales,
                                    base::cstring_view sizes,
                                    base::cstring_view mode,
                                    base::cstring_view output) {
  // Skip the input roi, which only takes effect when the coordinate
  // transformation mode is set to "tf_crop_and_resize". Currently WebNN only
  // supports "half_pixel", which is the default mode.
  const std::string roi;
  std::array<const char*, 4> inputs = {input.c_str(), roi.c_str(),
                                       scales.c_str(), sizes.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  std::array<ScopedOrtOpAttr, 1> attributes = {
      model_editor_.CreateAttribute(kAttrMode, mode)};

  model_editor_.AddNode(kOpTypeResize, node_name, inputs, outputs, attributes);
}

std::string GraphBuilderOrt::BlockwiseExpand(base::cstring_view input,
                                             base::span<const uint32_t> shape) {
  const std::string sizes = CreateInt64InitializerForUint32Array(shape);
  const std::string node_name = GenerateNodeName(
      base::JoinString({kInserted, kOpTypeResize}, kUnderscore));
  const std::string output = GenerateOperandName();
  AddResizeNode(node_name, input, /*scales=*/"", sizes,
                /*mode=*/"nearest", output);

  return output;
}

void GraphBuilderOrt::AddReshapeNode(base::cstring_view node_name,
                                     base::cstring_view input,
                                     base::cstring_view output,
                                     base::span<const uint32_t> shape) {
  // `new_shape` should be the name of an int64 tensor that specifies the
  // output's shape.
  const std::string new_shape = CreateInt64InitializerForUint32Array(shape);

  std::array<const char*, 2> inputs = {input.c_str(), new_shape.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  model_editor_.AddNode(kOpTypeReshape, node_name, inputs, outputs);
}

std::string GraphBuilderOrt::CreateReshapeNode(
    base::cstring_view input,
    base::span<const uint32_t> shape) {
  const std::string output = GenerateOperandName();
  InsertReshapeNode(input, output, shape);
  return output;
}

void GraphBuilderOrt::InsertReshapeNode(base::cstring_view input,
                                        base::cstring_view output,
                                        base::span<const uint32_t> shape) {
  const std::string node_name = GenerateNodeName(
      base::JoinString({kInserted, kOpTypeReshape}, kUnderscore));
  AddReshapeNode(node_name, input, output, shape);
}

void GraphBuilderOrt::AddSliceNode(base::cstring_view node_name,
                                   base::cstring_view input,
                                   base::cstring_view output,
                                   base::span<const int64_t> axes_value,
                                   base::span<const int64_t> starts_value,
                                   base::span<const int64_t> ends_value,
                                   base::span<const int64_t> steps_value) {
  // ONNX `Slice` op's `axes`, `starts`， `ends` and `steps` are operands of
  // data type int64 rather than attributes.
  const std::string axes = Create1DInitializer<int64_t>(axes_value);
  const std::string starts = Create1DInitializer<int64_t>(starts_value);
  const std::string ends = Create1DInitializer<int64_t>(ends_value);
  const std::string steps = Create1DInitializer<int64_t>(steps_value);

  std::array<const char*, 5> inputs = {
      input.c_str(), starts.c_str(), ends.c_str(), axes.c_str(), steps.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  model_editor_.AddNode(kOpTypeSlice, node_name, inputs, outputs);
}

void GraphBuilderOrt::AddTransposeNode(base::cstring_view node_name,
                                       base::cstring_view input,
                                       base::cstring_view output,
                                       base::span<const uint32_t> perm_value) {
  std::array<const char*, 1> inputs = {input.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  base::FixedArray<int64_t> perm(perm_value.begin(), perm_value.end());
  std::array<ScopedOrtOpAttr, 1> attributes = {
      model_editor_.CreateAttribute(kAttrPerm, perm)};
  model_editor_.AddNode(kOpTypeTranspose, node_name, inputs, outputs,
                        attributes);
}

std::string GraphBuilderOrt::CreateTransposeNode(
    base::cstring_view input,
    base::span<const uint32_t> perm_value) {
  const std::string node_name = GenerateNodeName(
      base::JoinString({kInserted, kOpTypeTranspose}, kUnderscore));
  const std::string output = GenerateOperandName();

  AddTransposeNode(node_name, input, output, perm_value);
  return output;
}

void GraphBuilderOrt::EmulateWithIdentityNode(base::cstring_view label,
                                              base::cstring_view input,
                                              base::cstring_view output) {
  const std::string node_name = GenerateNodeName(base::JoinString(
      {kInserted, kOpTypeIdentity, kToEmulate, label}, kUnderscore));

  std::array<const char*, 1> inputs = {input.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  model_editor_.AddNode(kOpTypeIdentity, node_name, inputs, outputs);
}

std::string GraphBuilderOrt::ClampIndices(base::cstring_view indices,
                                          OperandDataType data_type,
                                          uint32_t dim_size) {
  const std::string node_name = GenerateNodeName(
      base::JoinString({kInserted, kOpTypeClamp}, kUnderscore));
  const std::string output = GenerateOperandName();

  // The dimension size must be greater than 0.
  CHECK_GT(dim_size, 0u);

  std::string min;
  std::string max;
  switch (data_type) {
    case OperandDataType::kInt32: {
      // A valid dimension must be in the range of int32.
      // https://www.w3.org/TR/webnn/#valid-dimension
      min = CreateScalarInitializer(-base::checked_cast<int32_t>(dim_size));
      max = CreateScalarInitializer(base::checked_cast<int32_t>(dim_size - 1));
      break;
    }
    case OperandDataType::kInt64: {
      min = CreateScalarInitializer(-static_cast<int64_t>(dim_size));
      max = CreateScalarInitializer(static_cast<int64_t>(dim_size - 1));
      break;
    }
    default:
      NOTREACHED() << "[WebNN] Indices can only be one of the int32 and int64 "
                      "data types.";
  }

  std::array<const char*, 3> inputs = {indices.data(), min.c_str(),
                                       max.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  model_editor_.AddNode(kOpTypeClamp, node_name, inputs, outputs);
  return output;
}

std::string GraphBuilderOrt::ClampGatherNDIndices(
    base::cstring_view indices,
    base::span<const uint32_t> input_shape,
    base::span<const uint32_t> indices_shape) {
  CHECK_GT(input_shape.size(), 0u);
  CHECK_GT(indices_shape.size(), 0u);

  uint32_t indices_last_dim_size = indices_shape[indices_shape.size() - 1];
  std::array<int64_t, 1> min_max_shape = {
      static_cast<int64_t>(indices_last_dim_size)};

  base::FixedArray<int64_t> min_value(indices_last_dim_size);
  base::FixedArray<int64_t> max_value(indices_last_dim_size);
  for (uint32_t axis = 0; axis < indices_last_dim_size; ++axis) {
    min_value[axis] = -static_cast<int64_t>(input_shape[axis]);
    max_value[axis] = static_cast<int64_t>(input_shape[axis]) - 1;
  }

  // ONNX Clip can only have `min` and `max` as scalars, so here use Min and Max
  // to emulate a clamp operation.
  std::string min = CreateInitializer<int64_t>(min_max_shape, min_value);
  const std::string max_node_name =
      GenerateNodeName(base::JoinString({kInserted, kOpTypeMax}, kUnderscore));
  const std::string max_output = GenerateOperandName();
  std::array<const char*, 2> max_inputs = {indices.c_str(), min.c_str()};
  std::array<const char*, 1> max_outputs = {max_output.c_str()};
  model_editor_.AddNode(kOpTypeMax, max_node_name, max_inputs, max_outputs);

  std::string max = CreateInitializer<int64_t>(min_max_shape, max_value);
  const std::string min_node_name =
      GenerateNodeName(base::JoinString({kInserted, kOpTypeMin}, kUnderscore));
  const std::string output = GenerateOperandName();
  std::array<const char*, 2> min_inputs = {max_output.c_str(), max.c_str()};
  std::array<const char*, 1> min_outputs = {output.c_str()};
  model_editor_.AddNode(kOpTypeMin, min_node_name, min_inputs, min_outputs);

  return output;
}

template <typename T>
void GraphBuilderOrt::AddBinaryOperation(const T& operation,
                                         base::cstring_view op_type) {
  const std::string node_name = GenerateNodeName(operation.label);
  const std::string lhs = GetOperandNameById(operation.lhs_operand_id);
  const std::string rhs = GetOperandNameById(operation.rhs_operand_id);
  const std::string output = GetOperandNameById(operation.output_operand_id);

  std::array<const char*, 2> inputs = {lhs.c_str(), rhs.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  model_editor_.AddNode(op_type, node_name, inputs, outputs);
}

template <typename T>
void GraphBuilderOrt::AddUnaryOperation(const T& operation,
                                        base::cstring_view op_type) {
  const std::string node_name = GenerateNodeName(operation.label);
  const std::string input = GetOperandNameById(operation.input_operand_id);
  const std::string output = GetOperandNameById(operation.output_operand_id);

  std::array<const char*, 1> inputs = {input.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  model_editor_.AddNode(op_type, node_name, inputs, outputs);
}

void GraphBuilderOrt::AddArgMinMaxOperation(
    const mojom::ArgMinMax& arg_min_max) {
  const std::string node_name = GenerateNodeName(arg_min_max.label);
  const std::string input = GetOperandNameById(arg_min_max.input_operand_id);
  const std::string output = GetOperandNameById(arg_min_max.output_operand_id);

  CHECK(context_properties_.data_type_limits.arg_min_max_input.Supports(
      GetOperand(arg_min_max.input_operand_id).descriptor));
  CHECK(context_properties_.data_type_limits.arg_min_max_output.Supports(
      GetOperand(arg_min_max.output_operand_id).descriptor));

  std::array<ScopedOrtOpAttr, 2> attributes = {
      model_editor_.CreateAttribute(kAttrAxis,
                                    static_cast<int64_t>(arg_min_max.axis)),
      model_editor_.CreateAttribute(
          kAttrKeepDims, static_cast<int64_t>(arg_min_max.keep_dimensions))};

  // ONNX ArgMin/Max only supports int64 output.
  OperandDataType output_data_type =
      GetOperand(arg_min_max.output_operand_id).descriptor.data_type();
  bool need_cast = output_data_type != OperandDataType::kInt64;
  const std::string int64_output = need_cast ? GenerateOperandName() : output;

  std::array<const char*, 1> inputs = {input.c_str()};
  std::array<const char*, 1> outputs = {int64_output.c_str()};

  model_editor_.AddNode(arg_min_max.kind == mojom::ArgMinMax::Kind::kMax
                            ? kOpTypeArgMax
                            : kOpTypeArgMin,
                        node_name, inputs, outputs, attributes);

  if (need_cast) {
    // Here cast ArgMin/Max output from int64 to int32 is safe since WebNN
    // operand dimension must be in the range of int32.
    // https://www.w3.org/TR/webnn/#valid-dimension
    CHECK_EQ(output_data_type, OperandDataType::kInt32);
    InsertCastNode(int64_output, output, WebnnToOnnxDataType(output_data_type));
  }
}

void GraphBuilderOrt::AddBatchNormalizationOperation(
    const mojom::BatchNormalization& batch_normalization) {
  const std::string node_name = GenerateNodeName(batch_normalization.label);
  std::string input = GetOperandNameById(batch_normalization.input_operand_id);
  const std::string mean =
      GetOperandNameById(batch_normalization.mean_operand_id);
  const std::string variance =
      GetOperandNameById(batch_normalization.variance_operand_id);
  std::string output =
      GetOperandNameById(batch_normalization.output_operand_id);

  const DataTypeLimits& data_type_limits = context_properties_.data_type_limits;
  CHECK(data_type_limits.batch_normalization_input.Supports(
      GetOperand(batch_normalization.input_operand_id).descriptor));
  CHECK(data_type_limits.batch_normalization_mean.Supports(
      GetOperand(batch_normalization.mean_operand_id).descriptor));
  // TODO(crbug.com/431952809): Rename DataTypeLimits fields to be more generic
  // or encompassing.
  CHECK(data_type_limits.batch_normalization_mean.Supports(
      GetOperand(batch_normalization.variance_operand_id).descriptor));

  const OperandDescriptor& input_descriptor =
      GetOperand(batch_normalization.input_operand_id).descriptor;
  const OperandDataType input_data_type = input_descriptor.data_type();
  const std::vector<uint32_t>& input_shape = input_descriptor.shape();
  // ONNX BatchNormalization expects NCHW layout, channel is at index 1. In
  // addition it also accepts single dimension input of size N in which case C
  // is assumed to be 1.
  // https://onnx.ai/onnx/operators/onnx__BatchNormalization.html#inputs
  //
  // WebNN BatchNormalization supports 1D input of shape [C], but ONNX requires
  // at least 2D input. To handle this, we reshape [C] to [1, C] before passing
  // to ONNX, then reshape the output back to [C].
  bool needs_reshape_for_1d = input_shape.size() == 1;
  uint32_t input_channels = 1;

  if (needs_reshape_for_1d) {
    // Reshape 1D [C] -> 2D [1, C] for ONNX BatchNorm.

    input_channels = input_shape[0];
    input =
        CreateReshapeNode(input, {1, static_cast<uint32_t>(input_channels)});
  } else if (input_shape.size() > 1) {
    // For multi-dimensional inputs, channel is at index 1 (NCHW layout).
    input_channels = input_shape[1];
  }

  std::vector<uint32_t> scale_and_bias_shape = {input_channels};

  // ONNX BatchNormalization requires 5 inputs: input, scale, bias, mean and
  // variance. WebNN allows optional scale/bias, so create default ones if not
  // provided. Default scale = 1.0 (no scaling), default bias = 0.0 (no offset).
  std::string scale, bias;
  if (batch_normalization.scale_operand_id) {
    CHECK(data_type_limits.batch_normalization_mean.Supports(
        GetOperand(batch_normalization.scale_operand_id.value()).descriptor));
    scale = GetOperandNameById(batch_normalization.scale_operand_id.value());
  } else {
    scale = CreateOneInitializer(input_data_type, scale_and_bias_shape);
  }
  if (batch_normalization.bias_operand_id) {
    CHECK(data_type_limits.batch_normalization_mean.Supports(
        GetOperand(batch_normalization.bias_operand_id.value()).descriptor));
    bias = GetOperandNameById(batch_normalization.bias_operand_id.value());
  } else {
    bias = CreateZeroInitializer(input_data_type, scale_and_bias_shape);
  }

  // If we reshaped input from 1D to 2D, we need to reshape output back to 1D.
  std::string batchnorm_output = output;
  if (needs_reshape_for_1d) {
    batchnorm_output = GenerateOperandName();
  }

  std::array<const char*, 5> inputs = {input.c_str(), scale.c_str(),
                                       bias.c_str(), mean.c_str(),
                                       variance.c_str()};
  std::array<const char*, 1> outputs = {batchnorm_output.c_str()};
  std::array<ScopedOrtOpAttr, 1> attributes = {
      model_editor_.CreateAttribute(kAttrEpsilon, batch_normalization.epsilon)};
  model_editor_.AddNode(kOpTypeBatchNormalization, node_name, inputs, outputs,
                        attributes);

  // Reshape output back from 2D [1, C] -> 1D [C] for 1D inputs.
  if (needs_reshape_for_1d) {
    InsertReshapeNode(batchnorm_output, output,
                      {static_cast<uint32_t>(input_channels)});
  }
}

void GraphBuilderOrt::AddCastOperation(const mojom::ElementWiseUnary& cast) {
  const std::string node_name = GenerateNodeName(cast.label);
  const std::string input = GetOperandNameById(cast.input_operand_id);
  const std::string output = GetOperandNameById(cast.output_operand_id);
  const OperandDataType output_data_type =
      GetOperand(cast.output_operand_id).descriptor.data_type();
  AddCastNode(node_name, input, output, WebnnToOnnxDataType(output_data_type));
}

void GraphBuilderOrt::AddConv2dOperation(const mojom::Conv2d& conv2d) {
  const std::string node_name = GenerateNodeName(conv2d.label);
  const std::string input = GetOperandNameById(conv2d.input_operand_id);
  const std::string filter = GetOperandNameById(conv2d.filter_operand_id);
  const std::string output = GetOperandNameById(conv2d.output_operand_id);

  const DataTypeLimits& data_type_limits = context_properties_.data_type_limits;
  CHECK(data_type_limits.conv2d_input.Supports(
      GetOperand(conv2d.input_operand_id).descriptor));
  CHECK(data_type_limits.conv2d_input.Supports(
      GetOperand(conv2d.filter_operand_id).descriptor));
  std::vector<const char*> inputs = {input.c_str(), filter.c_str()};
  std::string bias;
  if (conv2d.bias_operand_id.has_value()) {
    CHECK(data_type_limits.conv2d_bias.Supports(
        GetOperand(conv2d.bias_operand_id.value()).descriptor));
    bias = GetOperandNameById(conv2d.bias_operand_id.value());
    inputs.push_back(bias.c_str());
  }
  std::array<const char*, 1> outputs = {output.c_str()};

  std::vector<ScopedOrtOpAttr> attributes;
  attributes.reserve(5);
  std::array<int64_t, 2> dilations = {
      base::checked_cast<int64_t>(conv2d.dilations->height),
      base::checked_cast<int64_t>(conv2d.dilations->width)};
  attributes.push_back(
      model_editor_.CreateAttribute(kAttrDilations, dilations));

  int64_t group = base::checked_cast<int64_t>(conv2d.groups);
  attributes.push_back(model_editor_.CreateAttribute(kAttrGroup, group));

  std::array<int64_t, 4> pads = {
      base::checked_cast<int64_t>(conv2d.padding->beginning->height),
      base::checked_cast<int64_t>(conv2d.padding->beginning->width),
      base::checked_cast<int64_t>(conv2d.padding->ending->height),
      base::checked_cast<int64_t>(conv2d.padding->ending->width)};
  attributes.push_back(model_editor_.CreateAttribute(kAttrPads, pads));

  std::array<int64_t, 2> strides = {
      base::checked_cast<int64_t>(conv2d.strides->height),
      base::checked_cast<int64_t>(conv2d.strides->width)};
  attributes.push_back(model_editor_.CreateAttribute(kAttrStrides, strides));

  switch (conv2d.kind) {
    case mojom::Conv2d::Kind::kDirect:
      model_editor_.AddNode(kOpTypeConv2d, node_name, inputs, outputs,
                            attributes);
      break;
    case mojom::Conv2d::Kind::kTransposed:
      // According to the ONNX ConvTranspose2d documentation, `output_padding`
      // is a zero vector if not specified and `pads` will be auto generated if
      // `output_shape` is specified. So we need to calculate the
      // `output_padding` and explicitly set it to ensure that the attributes
      // information is not missing. Since the `pads` attribute has already been
      // set, there is no need to set `output_size` attribute.
      // https://onnx.ai/onnx/operators/onnx__ConvTranspose.html#attributes
      const std::vector<uint32_t>& input_shape =
          GetOperand(conv2d.input_operand_id).descriptor.shape();
      const std::vector<uint32_t>& filter_shape =
          GetOperand(conv2d.filter_operand_id).descriptor.shape();
      const std::vector<uint32_t>& output_shape =
          GetOperand(conv2d.output_operand_id).descriptor.shape();
      // Since ONNX Runtime uses nchw input layout and oihw filter layout，
      // input/filter/output_shape[2] and input/filter/output_shape[3] are used
      // here to access the height and width dimensions of the
      // input/filter/output_shape tensor shape.
      std::array<int64_t, 2> input_size = {
          base::checked_cast<int64_t>(input_shape[2]),
          base::checked_cast<int64_t>(input_shape[3])};
      std::array<int64_t, 2> filter_size = {
          base::checked_cast<int64_t>(filter_shape[2]),
          base::checked_cast<int64_t>(filter_shape[3])};
      std::array<int64_t, 2> output_size = {
          base::checked_cast<int64_t>(output_shape[2]),
          base::checked_cast<int64_t>(output_shape[3])};

      int64_t output_padding_height = CalculateOutputPaddingSize(
          input_size[0], filter_size[0], strides[0], dilations[0], pads[0],
          pads[2], output_size[0]);
      int64_t output_padding_width = CalculateOutputPaddingSize(
          input_size[1], filter_size[1], strides[1], dilations[1], pads[1],
          pads[3], output_size[1]);
      std::array<int64_t, 2> output_padding = {output_padding_height,
                                               output_padding_width};

      attributes.push_back(
          model_editor_.CreateAttribute(kAttrOutputPadding, output_padding));

      model_editor_.AddNode(kOpTypeConvTranspose2d, node_name, inputs, outputs,
                            attributes);
      break;
  }
}

void GraphBuilderOrt::AddCumulativeSumOperation(
    const mojom::CumulativeSum& cumulative_sum) {
  const std::string node_name = GenerateNodeName(cumulative_sum.label);
  const std::string input = GetOperandNameById(cumulative_sum.input_operand_id);
  const std::string output =
      GetOperandNameById(cumulative_sum.output_operand_id);

  CHECK(context_properties_.data_type_limits.cumulative_sum_input.Supports(
      GetOperand(cumulative_sum.input_operand_id).descriptor));

  const std::string axis =
      CreateScalarInitializer(base::checked_cast<int64_t>(cumulative_sum.axis));

  std::array<ScopedOrtOpAttr, 2> attributes = {
      model_editor_.CreateAttribute(
          kAttrExclusive,
          base::checked_cast<int64_t>(cumulative_sum.exclusive)),
      model_editor_.CreateAttribute(
          kAttrReverse, base::checked_cast<int64_t>(cumulative_sum.reversed))};

  std::array<const char*, 2> inputs = {input.c_str(), axis.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};
  model_editor_.AddNode(kOpTypeCumulativeSum, node_name, inputs, outputs,
                        attributes);
}

template <typename T>
  requires(std::is_same_v<T, mojom::DequantizeLinear> ||
           std::is_same_v<T, mojom::QuantizeLinear>)
void GraphBuilderOrt::AddDequantizeOrQuantizeLinearOperation(
    const T& operation,
    base::cstring_view op_type) {
  const std::string node_name = GenerateNodeName(operation.label);
  std::string input = GetOperandNameById(operation.input_operand_id);
  std::string scale = GetOperandNameById(operation.scale_operand_id);
  std::string zero_point = GetOperandNameById(operation.zero_point_operand_id);
  std::string output = GetOperandNameById(operation.output_operand_id);

  const std::vector<uint32_t>& input_shape =
      GetOperand(operation.input_operand_id).descriptor.shape();
  // ZeroPoint has the same shape as the scale.
  const std::vector<uint32_t>& scale_zero_point_shape =
      GetOperand(operation.scale_operand_id).descriptor.shape();
  CHECK_EQ(scale_zero_point_shape.size(), input_shape.size());

  std::optional<int64_t> axis;
  uint32_t scale_not_size_one_dimension_count = 0;
  for (size_t i = 0; i < scale_zero_point_shape.size(); i++) {
    if (scale_zero_point_shape[i] != 1) {
      scale_not_size_one_dimension_count++;
      if (scale_zero_point_shape[i] == input_shape[i]) {
        axis = i;
      }
    }
  }

  bool is_per_axis =
      axis.has_value() && scale_not_size_one_dimension_count == 1;

  std::optional<int64_t> block_size;
  if (scale_not_size_one_dimension_count == 0) {
    // For per-tensor(per-layer) quantization and dequantization, scale should
    // be a scalar.
    if (!scale_zero_point_shape.empty()) {
      // The numbers in scale shape are all 1, scale and zeroPoint should be
      // reshaped to a scalar.
      scale = CreateReshapeNode(scale, {});
      zero_point = CreateReshapeNode(zero_point, {});
    }
  } else if (is_per_axis) {
    // For per-axis quantization and dequantization, scale and zeroPoint should
    // be a 1-D Tensor.
    if (scale_zero_point_shape.size() != 1) {
      scale = CreateReshapeNode(scale, {input_shape[axis.value()]});
      zero_point = CreateReshapeNode(zero_point, {input_shape[axis.value()]});
    }
  } else {
    // For blockwise quantization and dequantization, scale should has the same
    // shape as the input or except for one dimension in which blocking is
    // performed.
    // The default values are used if scale has the same shape as the input.
    axis = 0;
    block_size = 1;
    uint32_t blockwise_axis_count = 0;
    for (size_t i = 0; i < scale_zero_point_shape.size(); i++) {
      if (scale_zero_point_shape[i] != input_shape[i]) {
        CHECK_EQ(input_shape[i] % scale_zero_point_shape[i], 0u);
        block_size = input_shape[i] / scale_zero_point_shape[i];
        axis = i;
        blockwise_axis_count++;
      }
    }

    if (blockwise_axis_count > 1) {
      // The data type of zero point can be int4/uint4, which is not
      // supported by `resize` operator. So cast it to int8/uint8 before
      // `resize` and cast back to int4/uint4 after `resize`.
      const OperandDataType zero_point_data_type =
          GetOperand(operation.zero_point_operand_id).descriptor.data_type();
      if (zero_point_data_type == OperandDataType::kInt4) {
        zero_point =
            CreateCastNode(zero_point, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8);
      } else if (zero_point_data_type == OperandDataType::kUint4) {
        zero_point =
            CreateCastNode(zero_point, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
      }

      scale = BlockwiseExpand(scale, input_shape);
      zero_point = BlockwiseExpand(zero_point, input_shape);

      if (zero_point_data_type == OperandDataType::kInt4) {
        zero_point =
            CreateCastNode(zero_point, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4);
      } else if (zero_point_data_type == OperandDataType::kUint4) {
        zero_point =
            CreateCastNode(zero_point, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4);
      }

      // Reset the axis and block_size back to default values, because scale and
      // zeroPoint now have the same shape as input.
      axis = 0;
      block_size = 1;
    }
  }

  std::array<const char*, 3> inputs = {input.c_str(), scale.c_str(),
                                       zero_point.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  std::vector<ScopedOrtOpAttr> attributes;
  if (axis.has_value()) {
    attributes.push_back(
        model_editor_.CreateAttribute(kAttrAxis, axis.value()));
  }

  if (block_size.has_value()) {
    attributes.push_back(
        model_editor_.CreateAttribute(kAttrBlockSize, block_size.value()));
  }

  model_editor_.AddNode(op_type, node_name, inputs, outputs, attributes);
}

void GraphBuilderOrt::AddEluOperation(const mojom::Elu& elu) {
  const std::string node_name = GenerateNodeName(elu.label);
  const std::string input = GetOperandNameById(elu.input_operand_id);
  const std::string output = GetOperandNameById(elu.output_operand_id);

  CHECK(context_properties_.data_type_limits.elu_input.Supports(
      GetOperand(elu.input_operand_id).descriptor));

  std::array<ScopedOrtOpAttr, 1> attributes = {
      model_editor_.CreateAttribute(kAttrAlpha, elu.alpha)};

  std::array<const char*, 1> inputs = {input.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};
  model_editor_.AddNode(kOpTypeElu, node_name, inputs, outputs, attributes);
}

// TODO(crbug.com/426228071): Eliminate redundant cast ops for bool and uint8
// data types conversion.
void GraphBuilderOrt::AddLogicalBinaryOperation(
    const mojom::ElementWiseBinary& logical_binary,
    base::cstring_view op_type) {
  const std::string node_name = GenerateNodeName(logical_binary.label);
  std::string lhs = GetOperandNameById(logical_binary.lhs_operand_id);
  std::string rhs = GetOperandNameById(logical_binary.rhs_operand_id);

  // Some ONNX logical binary operations only support bool input.
  if (logical_binary.kind == mojom::ElementWiseBinary::Kind::kLogicalAnd ||
      logical_binary.kind == mojom::ElementWiseBinary::Kind::kLogicalOr ||
      logical_binary.kind == mojom::ElementWiseBinary::Kind::kLogicalXor) {
    CHECK_EQ(GetOperand(logical_binary.lhs_operand_id).descriptor.data_type(),
             OperandDataType::kUint8);
    lhs = CreateCastNode(lhs, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);

    CHECK_EQ(GetOperand(logical_binary.rhs_operand_id).descriptor.data_type(),
             OperandDataType::kUint8);
    rhs = CreateCastNode(rhs, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
  }
  std::array<const char*, 2> inputs = {lhs.c_str(), rhs.c_str()};

  const std::string bool_output = GenerateOperandName();
  std::array<const char*, 1> outputs = {bool_output.c_str()};
  model_editor_.AddNode(op_type, node_name, inputs, outputs);

  // ONNX logical operators only support bool output. WebNN logical operators
  // support uint8 output. It is necessary to insert a cast operator after a
  // logical operator.
  const OperandDataType output_data_type =
      GetOperand(logical_binary.output_operand_id).descriptor.data_type();
  const std::string output =
      GetOperandNameById(logical_binary.output_operand_id);
  CHECK_EQ(output_data_type, OperandDataType::kUint8);
  InsertCastNode(bool_output, output, WebnnToOnnxDataType(output_data_type));
}

void GraphBuilderOrt::AddLogicalUnaryOperation(
    const mojom::ElementWiseUnary& logical_unary,
    base::cstring_view op_type) {
  const std::string node_name = GenerateNodeName(logical_unary.label);

  std::string input = GetOperandNameById(logical_unary.input_operand_id);

  // LogicalNot operation in ONNX only supports bool input.
  if (op_type == kOpTypeLogicalNot) {
    CHECK_EQ(GetOperand(logical_unary.input_operand_id).descriptor.data_type(),
             OperandDataType::kUint8);
    input = CreateCastNode(input, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
  }

  const std::string bool_output = GenerateOperandName();

  std::array<const char*, 1> inputs = {input.c_str()};
  std::array<const char*, 1> outputs = {bool_output.c_str()};
  model_editor_.AddNode(op_type, node_name, inputs, outputs);

  // ONNX logical operators only support bool output, while WebNN logical
  // operators support uint8 output. Insert a `Cast` operator for type
  // conversion.
  const OperandDataType output_data_type =
      GetOperand(logical_unary.output_operand_id).descriptor.data_type();
  const std::string output =
      GetOperandNameById(logical_unary.output_operand_id);
  CHECK_EQ(output_data_type, OperandDataType::kUint8);
  InsertCastNode(bool_output, output, WebnnToOnnxDataType(output_data_type));
}

void GraphBuilderOrt::AddLogicalNotEqualOperation(
    const mojom::ElementWiseBinary& not_equal) {
  // Step 1: calculate `equal(a, b)`.
  const std::string equal_node_name =
      GenerateNodeName(GenerateEmulatedOpLabel(kOpTypeEqual, not_equal.label));
  std::string lhs = GetOperandNameById(not_equal.lhs_operand_id);
  std::string rhs = GetOperandNameById(not_equal.rhs_operand_id);
  const std::string equal_output = GenerateOperandName();

  std::array<const char*, 1> equal_outputs = {equal_output.c_str()};
  std::array<const char*, 2> equal_inputs = {lhs.c_str(), rhs.c_str()};
  model_editor_.AddNode(kOpTypeEqual, equal_node_name, equal_inputs,
                        equal_outputs);

  // Step 2: calculate `logicalNot(equal_output)`
  const std::string not_output = GenerateOperandName();
  std::array<const char*, 1> not_outputs = {not_output.c_str()};
  const std::string not_node_name = GenerateNodeName(
      GenerateEmulatedOpLabel(kOpTypeLogicalNot, not_equal.label));
  model_editor_.AddNode(kOpTypeLogicalNot, not_node_name, equal_outputs,
                        not_outputs);

  // ONNX logical operators only support bool output. To support output with the
  // WebNN data type, it is necessary to insert a cast operator after a logical
  // operator.
  OperandId output_operand_id = not_equal.output_operand_id;
  const OperandDataType output_data_type =
      GetOperand(output_operand_id).descriptor.data_type();
  std::string output = GetOperandNameById(output_operand_id);
  CHECK_EQ(output_data_type, OperandDataType::kUint8);
  InsertCastNode(not_output, output, WebnnToOnnxDataType(output_data_type));
}

void GraphBuilderOrt::AddElementWiseBinaryOperation(
    const mojom::ElementWiseBinary& element_wise_binary) {
  const DataTypeLimits& data_type_limits = context_properties_.data_type_limits;
  const OperandDescriptor& lhs_descriptor =
      GetOperand(element_wise_binary.lhs_operand_id).descriptor;
  const OperandDescriptor& rhs_descriptor =
      GetOperand(element_wise_binary.rhs_operand_id).descriptor;
  switch (element_wise_binary.kind) {
    case mojom::ElementWiseBinary::Kind::kAdd: {
      CHECK(data_type_limits.add_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddBinaryOperation(element_wise_binary, kOpTypeAdd);
    }
    case mojom::ElementWiseBinary::Kind::kSub: {
      CHECK(data_type_limits.sub_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddBinaryOperation(element_wise_binary, kOpTypeSub);
    }
    case mojom::ElementWiseBinary::Kind::kMul: {
      CHECK(data_type_limits.mul_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddBinaryOperation(element_wise_binary, kOpTypeMul);
    }
    case mojom::ElementWiseBinary::Kind::kDiv: {
      CHECK(data_type_limits.div_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddBinaryOperation(element_wise_binary, kOpTypeDiv);
    }
    case mojom::ElementWiseBinary::Kind::kMax: {
      CHECK(data_type_limits.max_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddBinaryOperation(element_wise_binary, kOpTypeMax);
    }
    case mojom::ElementWiseBinary::Kind::kMin: {
      CHECK(data_type_limits.min_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddBinaryOperation(element_wise_binary, kOpTypeMin);
    }
    case mojom::ElementWiseBinary::Kind::kPow: {
      CHECK(data_type_limits.pow_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddBinaryOperation(element_wise_binary, kOpTypePow);
    }
    case mojom::ElementWiseBinary::Kind::kEqual: {
      CHECK(data_type_limits.equal_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddLogicalBinaryOperation(element_wise_binary, kOpTypeEqual);
    }
    case mojom::ElementWiseBinary::Kind::kNotEqual: {
      CHECK(data_type_limits.not_equal_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddLogicalNotEqualOperation(element_wise_binary);
    }
    case mojom::ElementWiseBinary::Kind::kGreater: {
      CHECK(data_type_limits.greater_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddLogicalBinaryOperation(element_wise_binary, kOpTypeGreater);
    }
    case mojom::ElementWiseBinary::Kind::kGreaterOrEqual: {
      CHECK(data_type_limits.greater_or_equal_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddLogicalBinaryOperation(element_wise_binary,
                                       kOpTypeGreaterOrEqual);
    }
    case mojom::ElementWiseBinary::Kind::kLesser: {
      CHECK(data_type_limits.lesser_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddLogicalBinaryOperation(element_wise_binary, kOpTypeLesser);
    }
    case mojom::ElementWiseBinary::Kind::kLesserOrEqual: {
      CHECK(data_type_limits.lesser_or_equal_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddLogicalBinaryOperation(element_wise_binary,
                                       kOpTypeLesserOrEqual);
    }
    case mojom::ElementWiseBinary::Kind::kLogicalAnd: {
      CHECK(data_type_limits.logical_and_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddLogicalBinaryOperation(element_wise_binary, kOpTypeLogicalAnd);
    }
    case mojom::ElementWiseBinary::Kind::kLogicalOr: {
      CHECK(data_type_limits.logical_or_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddLogicalBinaryOperation(element_wise_binary, kOpTypeLogicalOr);
    }
    case mojom::ElementWiseBinary::Kind::kLogicalXor: {
      CHECK(data_type_limits.logical_xor_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddLogicalBinaryOperation(element_wise_binary, kOpTypeLogicalXor);
    }
  }
}

void GraphBuilderOrt::AddElementWiseUnaryOperation(
    const mojom::ElementWiseUnary& element_wise_unary) {
  const DataTypeLimits& data_type_limits = context_properties_.data_type_limits;
  const OperandDescriptor& input_descriptor =
      GetOperand(element_wise_unary.input_operand_id).descriptor;
  switch (element_wise_unary.kind) {
    case mojom::ElementWiseUnary::Kind::kAbs: {
      CHECK(data_type_limits.abs_input.Supports(input_descriptor));
      return AddUnaryOperation(element_wise_unary, kOpTypeAbs);
    }
    case mojom::ElementWiseUnary::Kind::kCeil: {
      CHECK(data_type_limits.ceil_input.Supports(input_descriptor));
      return AddUnaryOperation(element_wise_unary, kOpTypeCeil);
    }
    case mojom::ElementWiseUnary::Kind::kCos: {
      CHECK(data_type_limits.cos_input.Supports(input_descriptor));
      return AddUnaryOperation(element_wise_unary, kOpTypeCos);
    }
    case mojom::ElementWiseUnary::Kind::kExp: {
      CHECK(data_type_limits.exp_input.Supports(input_descriptor));
      return AddUnaryOperation(element_wise_unary, kOpTypeExp);
    }
    case mojom::ElementWiseUnary::Kind::kFloor: {
      CHECK(data_type_limits.floor_input.Supports(input_descriptor));
      return AddUnaryOperation(element_wise_unary, kOpTypeFloor);
    }
    case mojom::ElementWiseUnary::Kind::kLog: {
      CHECK(data_type_limits.log_input.Supports(input_descriptor));
      return AddUnaryOperation(element_wise_unary, kOpTypeLog);
    }
    case mojom::ElementWiseUnary::Kind::kIsNaN: {
      CHECK(data_type_limits.is_nan_input.Supports(input_descriptor));
      return AddLogicalUnaryOperation(element_wise_unary, kOpTypeIsNaN);
    }
    case mojom::ElementWiseUnary::Kind::kIsInfinite: {
      CHECK(data_type_limits.is_infinite_input.Supports(input_descriptor));
      return AddLogicalUnaryOperation(element_wise_unary, kOpTypeIsInfinite);
    }
    case mojom::ElementWiseUnary::Kind::kLogicalNot: {
      CHECK(data_type_limits.logical_not_input.Supports(input_descriptor));
      return AddLogicalUnaryOperation(element_wise_unary, kOpTypeLogicalNot);
    }
    case mojom::ElementWiseUnary::Kind::kNeg: {
      CHECK(data_type_limits.neg_input.Supports(input_descriptor));
      return AddUnaryOperation(element_wise_unary, kOpTypeNeg);
    }
    case mojom::ElementWiseUnary::Kind::kRoundEven: {
      CHECK(data_type_limits.round_even_input.Supports(input_descriptor));
      return AddUnaryOperation(element_wise_unary, kOpTypeRoundEven);
    }
    case mojom::ElementWiseUnary::Kind::kSign: {
      CHECK(data_type_limits.sign_input.Supports(input_descriptor));
      return AddUnaryOperation(element_wise_unary, kOpTypeSign);
    }
    case mojom::ElementWiseUnary::Kind::kSin: {
      CHECK(data_type_limits.sin_input.Supports(input_descriptor));
      return AddUnaryOperation(element_wise_unary, kOpTypeSin);
    }
    case mojom::ElementWiseUnary::Kind::kTan: {
      CHECK(data_type_limits.tan_input.Supports(input_descriptor));
      return AddUnaryOperation(element_wise_unary, kOpTypeTan);
    }
    case mojom::ElementWiseUnary::Kind::kIdentity: {
      CHECK(data_type_limits.identity_input.Supports(input_descriptor));
      return AddUnaryOperation(element_wise_unary, kOpTypeIdentity);
    }
    case mojom::ElementWiseUnary::Kind::kSqrt: {
      CHECK(data_type_limits.sqrt_input.Supports(input_descriptor));
      return AddUnaryOperation(element_wise_unary, kOpTypeSqrt);
    }
    case mojom::ElementWiseUnary::Kind::kErf: {
      CHECK(data_type_limits.erf_input.Supports(input_descriptor));
      return AddUnaryOperation(element_wise_unary, kOpTypeErf);
    }
    case mojom::ElementWiseUnary::Kind::kReciprocal: {
      CHECK(data_type_limits.reciprocal_input.Supports(input_descriptor));
      return AddUnaryOperation(element_wise_unary, kOpTypeReciprocal);
    }
    case mojom::ElementWiseUnary::Kind::kCast: {
      CHECK(data_type_limits.cast_input.Supports(input_descriptor));
      return AddCastOperation(element_wise_unary);
    }
  }
}

void GraphBuilderOrt::AddClampOperation(const mojom::Clamp& clamp) {
  const std::string node_name = GenerateNodeName(clamp.label);
  const std::string input = GetOperandNameById(clamp.input_operand_id);
  const std::string output = GetOperandNameById(clamp.output_operand_id);

  const DataTypeLimits& data_type_limits = context_properties_.data_type_limits;
  const OperandDescriptor& input_descriptor =
      GetOperand(clamp.input_operand_id).descriptor;
  CHECK(data_type_limits.clamp_input.Supports(input_descriptor));

  const OperandDataType input_data_type = input_descriptor.data_type();

  // Min and max are 0-D operands with the same data type of input.
  const std::string min =
      CreateScalarInitializer(input_data_type, clamp.min_value);
  const std::string max =
      CreateScalarInitializer(input_data_type, clamp.max_value);

  std::array<const char*, 3> inputs = {input.c_str(), min.c_str(), max.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  model_editor_.AddNode(kOpTypeClamp, node_name, inputs, outputs);
}

void GraphBuilderOrt::AddExpandOperation(const mojom::Expand& expand) {
  const std::string input = GetOperandNameById(expand.input_operand_id);
  const std::string output = GetOperandNameById(expand.output_operand_id);

  const OperandDescriptor& input_descriptor =
      GetOperand(expand.input_operand_id).descriptor;
  CHECK(context_properties_.data_type_limits.expand_input.Supports(
      input_descriptor));

  const OperandDescriptor& output_descriptor =
      GetOperand(expand.output_operand_id).descriptor;

  // Workaround: expanding a scalar to another scalar is supposed to be a no-op,
  // here we map it to an Identity node to avoid the mishandling of some ORT
  // EPs.
  // TODO(crbug.com/500385615): Remove the workaround when the issue is fixed.
  if (input_descriptor.Rank() == 0 && output_descriptor.Rank() == 0) {
    EmulateWithIdentityNode(expand.label, input, output);
    return;
  }

  const std::string node_name =