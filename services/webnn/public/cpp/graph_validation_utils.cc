// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/public/cpp/graph_validation_utils.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <set>
#include <variant>
#include <vector>

#include "base/check.h"
#include "base/check_op.h"
#include "base/notreached.h"
#include "base/numerics/checked_math.h"
#include "base/numerics/safe_conversions.h"
#include "base/strings/strcat.h"
#include "base/strings/string_util.h"
#include "base/strings/stringprintf.h"
#include "base/types/expected.h"
#include "base/types/expected_macros.h"
#include "base/types/fixed_array.h"
#include "services/webnn/public/cpp/context_properties.h"
#include "services/webnn/public/cpp/operand_descriptor.h"
#include "services/webnn/public/cpp/supported_data_types.h"
#include "services/webnn/public/cpp/supported_tensors.h"
#include "services/webnn/public/cpp/webnn_errors.h"

namespace webnn {

namespace {

struct Conv2dInputOutputInfo {
  uint32_t batches;
  uint32_t channels;
  uint32_t height;
  uint32_t width;
};

// The error message labels for corresponding operands.
static constexpr char kBiasParam[] = "bias";
static constexpr char kCellStateParam[] = "cellState";
static constexpr char kConditionParam[] = "condition";
static constexpr char kFalseValueParam[] = "falseValue";
static constexpr char kFilterParam[] = "filter";
static constexpr char kGemmAParam[] = "gemmA";
static constexpr char kGemmBParam[] = "gemmB";
static constexpr char kGemmCParam[] = "gemmC";
static constexpr char kHiddenStateParam[] = "hiddenState";
static constexpr char kIndicesParam[] = "indices";
static constexpr char kInitialCellStateParam[] = "initialCellState";
static constexpr char kInitialHiddenStateParam[] = "initialHiddenState";
static constexpr char kMeanParam[] = "mean";
static constexpr char kPeepholeWeightParam[] = "peepholeWeight";
static constexpr char kRecurrentBiasParam[] = "recurrentBias";
static constexpr char kRecurrentWeightParam[] = "recurrentWeight";
static constexpr char kScaleParam[] = "scale";
static constexpr char kSlopeParam[] = "slope";
static constexpr char kTrueValueParam[] = "trueValue";
static constexpr char kUpdatesParam[] = "updates";
static constexpr char kVarianceParam[] = "variance";
static constexpr char kWeightParam[] = "weight";
static constexpr char kZeroPointParam[] = "zeroPoint";

// Validate that the intermediate padded shape is within the limits of
// OperandDescriptor. This is useful for convolution and pooling operations
// that may be implemented by padding the input tensor first.
base::expected<void, std::string> ValidateIntermediatePaddedDescriptor(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    const Padding2d& padding,
    const InputOperandLayout& input_layout,
    const Conv2dInputOutputInfo& input_info,
    std::string_view label) {
  uint32_t padded_height;
  uint32_t padded_width;
  if (!(base::CheckedNumeric<uint32_t>(input_info.height) +
        padding.beginning.height + padding.ending.height)
           .AssignIfValid(&padded_height) ||
      !(base::CheckedNumeric<uint32_t>(input_info.width) +
        padding.beginning.width + padding.ending.width)
           .AssignIfValid(&padded_width)) {
    return base::unexpected(
        ErrorWithLabel(label, "The padded intermediate shape is too large."));
  }

  std::array<uint32_t, 4> padded_shape;
  switch (input_layout) {
    case InputOperandLayout::kNchw:
      padded_shape = {input_info.batches, input_info.channels, padded_height,
                      padded_width};
      break;
    case InputOperandLayout::kNhwc:
      padded_shape = {input_info.batches, padded_height, padded_width,
                      input_info.channels};
      break;
  }

  auto padded_descriptor = OperandDescriptor::Create(
      context_properties, input.data_type(), padded_shape, label);
  if (!padded_descriptor.has_value()) {
    return base::unexpected(ErrorWithLabel(
        label, base::StrCat({"The padded intermediate operand is invalid: ",
                             padded_descriptor.error()})));
  }

  return base::ok();
}

// Validate and calculate the output spatial dimensions of convTranspose2d given
// input sizes, filter sizes, padding, strides, dilations and output padding.
base::expected<Size2d<uint32_t>, std::string>
ValidateAndCalculateConvTranspose2dOutputSizes(
    const uint32_t input_height,
    const uint32_t input_width,
    const uint32_t filter_height,
    const uint32_t filter_width,
    const Padding2d& padding,
    const Size2d<uint32_t>& strides,
    const Size2d<uint32_t>& dilations,
    const Size2d<uint32_t>& output_padding,
    std::string_view label) {
  if (strides.height == 0 || strides.width == 0) {
    return base::unexpected(
        ErrorWithLabel(label, "All strides should be greater than 0."));
  }
  if (dilations.height == 0 || dilations.width == 0) {
    return base::unexpected(
        ErrorWithLabel(label, "All dilations should be greater than 0."));
  }
  if (output_padding.height >= strides.height ||
      output_padding.width >= strides.width) {
    return base::unexpected(ErrorWithLabel(
        label,
        "The output padding must be smaller than the stride along the same "
        "dimension."));
  }

  const auto output_height = CalculateConvTranspose2dOutputSize(
      input_height, filter_height, padding.beginning.height,
      padding.ending.height, strides.height, dilations.height,
      output_padding.height);
  if (!output_height.has_value()) {
    return base::unexpected(ErrorWithLabel(
        label,
        "Failed to calculate the output height: " + output_height.error()));
  }

  const auto output_width = CalculateConvTranspose2dOutputSize(
      input_width, filter_width, padding.beginning.width, padding.ending.width,
      strides.width, dilations.width, output_padding.width);
  if (!output_width.has_value()) {
    return base::unexpected(ErrorWithLabel(
        label,
        "Failed to calculate the output width: " + output_width.error()));
  }

  return Size2d<uint32_t>{.height = output_height.value(),
                          .width = output_width.value()};
}

// Get the input info of 2-D direct and transposed convolution
// operation given input operand and attributes.
Conv2dInputOutputInfo GetConv2dInputInfo(
    const std::string& label,
    const OperandDescriptor& input,
    const Conv2dAttributesBase& attributes) {
  const std::vector<uint32_t>& input_shape = input.shape();
  // The input layout option specifies the layout format of the input tensor.
  uint32_t batches, channels, height, width;
  switch (attributes.input_layout) {
    case InputOperandLayout::kNchw:
      // "nchw": [batches, input_channels, height, width]
      batches = input_shape[0];
      channels = input_shape[1];
      height = input_shape[2];
      width = input_shape[3];
      break;
    case InputOperandLayout::kNhwc:
      // "nhwc": [batches, height, width, input_channels]
      batches = input_shape[0];
      height = input_shape[1];
      width = input_shape[2];
      channels = input_shape[3];
      break;
  }

  return Conv2dInputOutputInfo{.batches = batches,
                               .channels = channels,
                               .height = height,
                               .width = width};
}

// Validate the bias of 2-D direct and transposed convolution operation and
// create output operand given input operand, attributes and output info.
base::expected<OperandDescriptor, std::string>
ValidateConv2dBiasAndCreateOutputOperand(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    const Conv2dAttributesBase& attributes,
    const Conv2dInputOutputInfo& output_info) {
  const std::string& label = attributes.label;
  // Validate bias operand if it is present.
  if (attributes.bias_operand) {
    if (attributes.bias_operand->shape()[0] != output_info.channels) {
      return base::unexpected(ErrorWithLabel(
          label, base::StringPrintf("The bias shape should be [%u].",
                                    output_info.channels)));
    }
    if (attributes.bias_operand->data_type() != input.data_type()) {
      return base::unexpected(ErrorWithLabel(
          label, "The bias data type doesn't match input data type."));
    }
  }

  // The input layout option specifies the layout format of the output tensor.
  std::array<uint32_t, 4> output_shape;
  switch (attributes.input_layout) {
    case InputOperandLayout::kNchw:
      // "nchw": [batches, output_channels, height, width]
      output_shape = {output_info.batches, output_info.channels,
                      output_info.height, output_info.width};
      break;
    case InputOperandLayout::kNhwc:
      // "nhwc": [batches, height, width, output_channels]
      output_shape = {output_info.batches, output_info.height,
                      output_info.width, output_info.channels};
      break;
  }

  return OperandDescriptor::Create(context_properties, input.data_type(),
                                   output_shape, label);
}

// Validate the axes and infer output for reduce operations.
base::expected<std::vector<uint32_t>, std::string>
ValidateReduceAxesAndInferOutput(base::span<const uint32_t> input_dimensions,
                                 base::span<const uint32_t> axes,
                                 bool keep_dimensions,
                                 std::string_view label) {
  auto input_rank = static_cast<uint32_t>(input_dimensions.size());
  RETURN_IF_ERROR(ValidateAxes(axes, input_rank, label));

  std::vector<uint32_t> output_shape;
  if (keep_dimensions) {
    output_shape.assign(input_dimensions.begin(), input_dimensions.end());
    for (auto axis : axes) {
      output_shape[axis] = 1;
    }
  } else {
    for (size_t i = 0; i < input_rank; i++) {
      if (!std::ranges::contains(axes, i)) {
        output_shape.push_back(input_dimensions[i]);
      }
    }
  }
  return output_shape;
}

// Validate the operand of recurrent network.
base::expected<void, std::string> ValidateRecurrentNetworkOperand(
    const OperandDescriptor& operand,
    const char* operand_name,
    base::span<const uint32_t> expected_shape,
    OperandDataType input_data_type,
    std::string_view label) {
  if (!std::ranges::equal(operand.shape(), expected_shape)) {
    return base::unexpected(ErrorWithLabel(
        label,
        base::StringPrintf("The %s operand shape is invalid.", operand_name)));
  }
  if (operand.data_type() != input_data_type) {
    return base::unexpected(ErrorWithLabel(
        label,
        base::StringPrintf(
            "The %s operand data type doesn't match the input data type.",
            operand_name)));
  }
  return base::ok();
}

// This helper method is intended to validate mean, variance, scale and bias
// operands of batchNormalization and instanceNormalization against the input
// operand. These operands share the same constraint.
base::expected<void, std::string>
ValidateNormalizationOperandIsCompatibleWithInput(
    const OperandDescriptor& operand,
    const OperandDataType input_data_type,
    size_t input_size_on_axis,
    std::string_view label,
    std::string_view argument_name) {
  if (operand.data_type() != input_data_type) {
    return base::unexpected(ErrorWithLabel(
        label,
        base::StrCat(
            {"For ", argument_name,
             " operand: the data type doesn't match the input data type."})));
  }

  if (operand.shape()[0] != input_size_on_axis) {
    return base::unexpected(ErrorWithLabel(
        label,
        base::StrCat({"For ", argument_name,
                      " operand: the size of operand must be equal to the size "
                      "of the feature dimension of the input."})));
  }

  return base::ok();
}

}  // namespace

base::expected<double, std::string> CalculateConv2dOutputSize(
    uint32_t input_size,
    uint32_t filter_size,
    uint32_t beginning_padding,
    uint32_t ending_padding,
    uint32_t stride,
    uint32_t dilation,
    std::string_view label) {
  // Calculate the dilated filter sizes.
  auto checked_effective_filter_size =
      (base::CheckedNumeric<uint32_t>(filter_size) - 1) * dilation + 1;
  if (!checked_effective_filter_size.IsValid()) {
    return base::unexpected(
        ErrorWithLabel(label, "The effective filter size is too large."));
  }

  // Calculate the output size in double precision floating point number that
  // ensures all dimension values of type uint32_t can be exactly represented.
  // https://en.wikipedia.org/wiki/Double-precision_floating-point_format#Precision_limitations_on_integer_values
  // The max value of checked_output_size should be 3 * UINT_MAX + 1,
  // which is smaller than the max safe integer value for double type.
  auto checked_output_size =
      (base::CheckedNumeric<double>(input_size) -
       checked_effective_filter_size + beginning_padding + ending_padding) /
          stride +
      1;

  if (checked_output_size.ValueOrDie() <= 0) {
    return base::unexpected(ErrorWithLabel(
        label, "The input size is too small to fill the window."));
  }

  // Check if the value is valid for rounding to uint32_t type.
  if (!checked_output_size.IsValid<uint32_t>()) {
    return base::unexpected(
        ErrorWithLabel(label, "The output size is too large."));
  }

  return checked_output_size.ValueOrDie();
}

BatchNormalizationAttributes::BatchNormalizationAttributes() = default;
BatchNormalizationAttributes::~BatchNormalizationAttributes() = default;
BatchNormalizationAttributes::BatchNormalizationAttributes(
    BatchNormalizationAttributes&& other) = default;
BatchNormalizationAttributes& BatchNormalizationAttributes::operator=(
    BatchNormalizationAttributes&& other) = default;

Conv2dAttributesBase::Conv2dAttributesBase() = default;
Conv2dAttributesBase::~Conv2dAttributesBase() = default;
Conv2dAttributesBase::Conv2dAttributesBase(Conv2dAttributesBase&& other) =
    default;
Conv2dAttributesBase& Conv2dAttributesBase::operator=(
    Conv2dAttributesBase&& other) = default;

Conv2dAttributes::Conv2dAttributes() = default;
Conv2dAttributes::~Conv2dAttributes() = default;
Conv2dAttributes::Conv2dAttributes(Conv2dAttributes&& other) = default;
Conv2dAttributes& Conv2dAttributes::operator=(Conv2dAttributes&& other) =
    default;

ConvTranspose2dAttributes::ConvTranspose2dAttributes() = default;
ConvTranspose2dAttributes::~ConvTranspose2dAttributes() = default;
ConvTranspose2dAttributes::ConvTranspose2dAttributes(
    ConvTranspose2dAttributes&& other) = default;
ConvTranspose2dAttributes& ConvTranspose2dAttributes::operator=(
    ConvTranspose2dAttributes&& other) = default;

GemmAttributes::GemmAttributes() = default;
GemmAttributes::~GemmAttributes() = default;
GemmAttributes::GemmAttributes(GemmAttributes&& other) = default;
GemmAttributes& GemmAttributes::operator=(GemmAttributes&& other) = default;

GruAttributes::GruAttributes() = default;
GruAttributes::~GruAttributes() = default;
GruAttributes::GruAttributes(GruAttributes&& other) = default;
GruAttributes& GruAttributes::operator=(GruAttributes&& other) = default;

GruCellAttributes::GruCellAttributes() = default;
GruCellAttributes::~GruCellAttributes() = default;
GruCellAttributes::GruCellAttributes(GruCellAttributes&& other) = default;
GruCellAttributes& GruCellAttributes::operator=(GruCellAttributes&& other) =
    default;

InstanceNormalizationAttributes::InstanceNormalizationAttributes() = default;
InstanceNormalizationAttributes::~InstanceNormalizationAttributes() = default;
InstanceNormalizationAttributes::InstanceNormalizationAttributes(
    InstanceNormalizationAttributes&& other) = default;
InstanceNormalizationAttributes& InstanceNormalizationAttributes::operator=(
    InstanceNormalizationAttributes&& other) = default;

LayerNormalizationAttributes::LayerNormalizationAttributes() = default;
LayerNormalizationAttributes::~LayerNormalizationAttributes() = default;
LayerNormalizationAttributes::LayerNormalizationAttributes(
    LayerNormalizationAttributes&& other) = default;
LayerNormalizationAttributes& LayerNormalizationAttributes::operator=(
    LayerNormalizationAttributes&& other) = default;

LstmAttributes::LstmAttributes() = default;
LstmAttributes::~LstmAttributes() = default;
LstmAttributes::LstmAttributes(LstmAttributes&& other) = default;
LstmAttributes& LstmAttributes::operator=(LstmAttributes&& other) = default;

LstmCellAttributes::LstmCellAttributes() = default;
LstmCellAttributes::~LstmCellAttributes() = default;
LstmCellAttributes::LstmCellAttributes(LstmCellAttributes&& other) = default;
LstmCellAttributes& LstmCellAttributes::operator=(LstmCellAttributes&& other) =
    default;

Pool2dAttributes::Pool2dAttributes() = default;
Pool2dAttributes::~Pool2dAttributes() = default;
Pool2dAttributes::Pool2dAttributes(Pool2dAttributes&& other) = default;
Pool2dAttributes& Pool2dAttributes::operator=(Pool2dAttributes&& other) =
    default;

SliceAttributes::SliceAttributes() = default;
SliceAttributes::~SliceAttributes() = default;
SliceAttributes::SliceAttributes(SliceAttributes&& other) = default;
SliceAttributes& SliceAttributes::operator=(SliceAttributes&& other) = default;

base::expected<OperandDescriptor, std::string> ValidateArgMinMaxAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    std::string_view label,
    uint32_t axis,
    OperandDataType output_data_type,
    bool keep_dimensions) {
  if (!context_properties.data_type_limits.arg_min_max_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedInputArgumentError(
            input, context_properties.data_type_limits.arg_min_max_input)));
  }

  if (!context_properties.data_type_limits.arg_min_max_output.data_types.Has(
          output_data_type)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedOpOutputTypeError(
                   output_data_type, context_properties.data_type_limits
                                         .arg_min_max_output.data_types)));
  }

  ASSIGN_OR_RETURN(std::vector<uint32_t> output_shape,
                   ValidateReduceAxesAndInferOutput(
                       input.shape(), std::array<uint32_t, 1>{axis},
                       keep_dimensions, label));

  return OperandDescriptor::Create(context_properties, output_data_type,
                                   output_shape, label);
}

base::expected<OperandDescriptor, std::string>
ValidateBatchNormalizationAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    const OperandDescriptor& mean,
    const OperandDescriptor& variance,
    const BatchNormalizationAttributes& attributes) {
  // Validate input operand.
  const std::string& label = attributes.label;
  if (!context_properties.data_type_limits.batch_normalization_input.Supports(
          input)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedInputArgumentError(
            input,
            context_properties.data_type_limits.batch_normalization_input)));
  }

  if (attributes.axis >= input.Rank()) {
    return base::unexpected(ErrorWithLabel(
        label,
        "The value of axis must be in the range [0, N-1] where N is the rank "
        "of the input tensor."));
  }

  uint32_t input_size_on_axis = input.shape()[attributes.axis];
  // Validate mean operand.
  if (!context_properties.data_type_limits.batch_normalization_mean.Supports(
          mean)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedArgumentError(
            kMeanParam, mean,
            context_properties.data_type_limits.batch_normalization_mean)));
  }
  RETURN_IF_ERROR(ValidateNormalizationOperandIsCompatibleWithInput(
      mean, input.data_type(), input_size_on_axis, label, kMeanParam));

  // Validate variance operand.
  if (!context_properties.data_type_limits.batch_normalization_mean.Supports(
          variance)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedArgumentError(
            kVarianceParam, variance,
            context_properties.data_type_limits.batch_normalization_mean)));
  }
  RETURN_IF_ERROR(ValidateNormalizationOperandIsCompatibleWithInput(
      variance, input.data_type(), input_size_on_axis, label, kVarianceParam));

  // Validate scale operand.
  if (attributes.scale) {
    if (!context_properties.data_type_limits.batch_normalization_mean.Supports(
            attributes.scale.value())) {
      return base::unexpected(ErrorWithLabel(
          label,
          NotSupportedArgumentError(
              kScaleParam, attributes.scale.value(),
              context_properties.data_type_limits.batch_normalization_mean)));
    }
    RETURN_IF_ERROR(ValidateNormalizationOperandIsCompatibleWithInput(
        attributes.scale.value(), input.data_type(), input_size_on_axis, label,
        kScaleParam));
  }

  // Validate bias operand.
  if (attributes.bias) {
    if (!context_properties.data_type_limits.batch_normalization_mean.Supports(
            attributes.bias.value())) {
      return base::unexpected(ErrorWithLabel(
          label,
          NotSupportedArgumentError(
              kBiasParam, attributes.bias.value(),
              context_properties.data_type_limits.batch_normalization_mean)));
    }
    RETURN_IF_ERROR(ValidateNormalizationOperandIsCompatibleWithInput(
        attributes.bias.value(), input.data_type(), input_size_on_axis, label,
        kBiasParam));
  }

  // The output tensor of batchNormalization is the same shape as the input
  // tensor.
  return input;
}

base::expected<OperandDescriptor, std::string> ValidateCastAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    OperandDataType output_data_type,
    std::string_view label) {
  // Validate input operand.
  if (!context_properties.data_type_limits.cast_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedInputArgumentError(
                   input, context_properties.data_type_limits.cast_input)));
  }

  // Validate output data type.
  if (!context_properties.data_type_limits.cast_input.data_types.Has(
          output_data_type)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedOpOutputTypeError(
                   output_data_type,
                   context_properties.data_type_limits.cast_input.data_types)));
  }

  return OperandDescriptor::Create(context_properties, output_data_type,
                                   input.shape(), label);
}

base::expected<OperandDescriptor, std::string> ValidateConcatAndInferOutput(
    const ContextProperties& context_properties,
    const std::vector<OperandDescriptor>& inputs,
    const uint32_t axis,
    std::string_view label) {
  if (inputs.empty()) {
    return base::unexpected(
        ErrorWithLabel(label, "The inputs should not be empty."));
  }

  if (inputs.size() > kMaxValidTensorCount) {
    return base::unexpected(ErrorWithLabel(
        label, base::StringPrintf(
                   "The number of inputs must be less than or equal to %u.",
                   kMaxValidTensorCount)));
  }

  for (const auto& input : inputs) {
    if (!context_properties.data_type_limits.concat_inputs.Supports(input)) {
      return base::unexpected(ErrorWithLabel(
          label,
          NotSupportedInputArgumentError(
              input, context_properties.data_type_limits.concat_inputs)));
    }
  }

  const auto first_input_rank = inputs[0].Rank();
  // According to WebNN spec:
  // https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-concat-inputs-axis-axis,
  // the axis that the inputs concatenate along, with the value in the interval
  // [0, N-1] where N is the rank of input tensors. We just check the first
  // input rank here because we will check all inputs have same rank in the
  // following loop.
  if (axis >= first_input_rank) {
    return base::unexpected(ErrorWithLabel(
        label,
        "The axis must be in the range [0, N-1] where N is the rank of input "
        "tensor."));
  }

  const std::vector<uint32_t>& first_input_shape = inputs[0].shape();
  const auto output_type = inputs[0].data_type();
  // The loop skips the first input to avoid repeated checks.
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (inputs[i].data_type() != output_type) {
      return base::unexpected(
          ErrorWithLabel(label, "The input data types don't match."));
    }
    // According to WebNN spec:
    // https://www.w3.org/TR/webnn/#api-mlgraphbuilder-concat, all input tensors
    // must have the same dimension.
    if (inputs[i].Rank() != first_input_rank) {
      return base::unexpected(ErrorWithLabel(
          label, "All input tensors must have the same dimension."));
    }
    // According to WebNN spec:
    // https://www.w3.org/TR/webnn/#api-mlgraphbuilder-concat, all input tensors
    // must have the same shape, except for the size of the dimension to
    // concatenate on.
    for (size_t dim = 0; dim < first_input_rank; ++dim) {
      if (dim == axis || inputs[i].shape()[dim] == first_input_shape[dim]) {
        continue;
      }
      return base::unexpected(ErrorWithLabel(
          label,
          "All input tensors must have the same shape, except for the size of "
          "the dimension to concatenate on."));
    }
  }
  // Calculate the output shape according to WebNN spec:
  // https://www.w3.org/TR/webnn/#api-mlgraphbuilder-concat, the output tensor
  // has the same shape except on the dimension that all the inputs concatenated
  // along. The size of that dimension is computed as the sum of all the input
  // sizes of the same dimension.
  auto axis_size = base::CheckedNumeric<uint32_t>(0);
  for (auto& input : inputs) {
    axis_size += input.shape()[axis];
  }
  std::vector<uint32_t> output_shape = first_input_shape;
  if (!axis_size.AssignIfValid(&output_shape[axis])) {
    return base::unexpected(
        ErrorWithLabel(label, "The concatenated dimension size is too large."));
  }

  return OperandDescriptor::Create(context_properties, output_type,
                                   output_shape, label);
}

base::expected<Size2d<double>, std::string>
ValidateAndCalculateConv2dOutputSizes(uint32_t input_height,
                                      uint32_t input_width,
                                      uint32_t filter_height,
                                      uint32_t filter_width,
                                      const Padding2d& padding,
                                      const Size2d<uint32_t>& strides,
                                      const Size2d<uint32_t>& dilations,
                                      std::string_view label) {
  if (strides.height == 0 || strides.width == 0) {
    return base::unexpected(
        ErrorWithLabel(label, "All strides should be greater than 0."));
  }
  if (dilations.height == 0 || dilations.width == 0) {
    return base::unexpected(
        ErrorWithLabel(label, "All dilations should be greater than 0."));
  }

  const auto float_output_height = CalculateConv2dOutputSize(
      input_height, filter_height, padding.beginning.height,
      padding.ending.height, strides.height, dilations.height, label);
  if (!float_output_height.has_value()) {
    return base::unexpected(
        ErrorWithLabel(label, "Failed to calculate the output height: " +
                                  float_output_height.error()));
  }

  const auto float_output_width = CalculateConv2dOutputSize(
      input_width, filter_width, padding.beginning.width, padding.ending.width,
      strides.width, dilations.width, label);
  if (!float_output_width.has_value()) {
    return base::unexpected(ErrorWithLabel(
        label,
        "Failed to calculate the output width: " + float_output_width.error()));
  }

  return Size2d<double>{.height = float_output_height.value(),
                        .width = float_output_width.value()};
}

base::expected<OperandDescriptor, std::string> ValidateConv2dAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    const OperandDescriptor& filter,
    const Conv2dAttributes& attributes) {
  const std::string& label = attributes.label;
  // Validate input operand.
  if (!context_properties.data_type_limits.conv2d_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedInputArgumentError(
                   input, context_properties.data_type_limits.conv2d_input)));
  }
  Conv2dInputOutputInfo input_info =
      GetConv2dInputInfo(label, input, attributes);

  // Validate filter operand.
  if (!context_properties.data_type_limits.conv2d_input.Supports(filter)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kFilterParam, filter,
                   context_properties.data_type_limits.conv2d_input)));
  }
  if (filter.data_type() != input.data_type()) {
    return base::unexpected(ErrorWithLabel(
        label, "The filter data type doesn't match the input data type."));
  }

  // Validate bias operand if it is present.
  if (attributes.bias_operand) {
    if (!context_properties.data_type_limits.conv2d_bias.Supports(
            attributes.bias_operand.value())) {
      return base::unexpected(ErrorWithLabel(
          label, NotSupportedArgumentError(
                     kBiasParam, attributes.bias_operand.value(),
                     context_properties.data_type_limits.conv2d_bias)));
    }
  }

  const std::vector<uint32_t>& filter_shape = filter.shape();
  uint32_t filter_height, filter_width, output_channels, filter_input_channels;
  // The conv2d filter layout specifies the filter layout format.
  switch (attributes.filter_layout) {
    case Conv2dFilterOperandLayout::kHwio:
      // "hwio": [height, width, input_channels/groups, output_channels]
      filter_height = filter_shape[0];
      filter_width = filter_shape[1];
      filter_input_channels = filter_shape[2];
      output_channels = filter_shape[3];
      break;
    case Conv2dFilterOperandLayout::kOhwi:
      // "ohwi": [output_channels, height, width, input_channels/groups]
      output_channels = filter_shape[0];
      filter_height = filter_shape[1];
      filter_width = filter_shape[2];
      filter_input_channels = filter_shape[3];
      break;
    case Conv2dFilterOperandLayout::kIhwo:
      // "ihwo": [input_channels/groups, height, width, output_channels]
      filter_input_channels = filter_shape[0];
      filter_height = filter_shape[1];
      filter_width = filter_shape[2];
      output_channels = filter_shape[3];
      break;
    case Conv2dFilterOperandLayout::kOihw:
      // "oihw": [output_channels, input_channels/groups, height, width]
      output_channels = filter_shape[0];
      filter_input_channels = filter_shape[1];
      filter_height = filter_shape[2];
      filter_width = filter_shape[3];
      break;
  }

  // Validate groups and input channels.
  if (attributes.groups == 0) {
    return base::unexpected(
        ErrorWithLabel(label, "The groups should be greater than 0."));
  }
  if (input_info.channels % attributes.groups != 0 ||
      filter_input_channels != input_info.channels / attributes.groups) {
    return base::unexpected(ErrorWithLabel(
        label,
        "The groups must evenly divide the input channels to filter input "
        "channels."));
  }
  if (output_channels % attributes.groups != 0) {
    return base::unexpected(ErrorWithLabel(
        label, "The groups must evenly divide the output channels."));
  }

  RETURN_IF_ERROR(ValidateIntermediatePaddedDescriptor(
      context_properties, input, attributes.padding, attributes.input_layout,
      input_info, label));

  // Validate and calculate output sizes.
  ASSIGN_OR_RETURN(
      Size2d<double> output_sizes,
      ValidateAndCalculateConv2dOutputSizes(
          input_info.height, input_info.width, filter_height, filter_width,
          attributes.padding, attributes.strides, attributes.dilations, label));

  uint32_t output_height = base::ClampFloor<uint32_t>(output_sizes.height);
  uint32_t output_width = base::ClampFloor<uint32_t>(output_sizes.width);

  Conv2dInputOutputInfo output_info{.batches = input_info.batches,
                                    .channels = output_channels,
                                    .height = output_height,
                                    .width = output_width};
  return ValidateConv2dBiasAndCreateOutputOperand(context_properties, input,
                                                  attributes, output_info);
}

base::expected<OperandDescriptor, std::string>
ValidateConvTranspose2dAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    const OperandDescriptor& filter,
    const ConvTranspose2dAttributes& attributes) {
  // Validate input operand.
  const std::string& label = attributes.label;
  if (!context_properties.data_type_limits.conv_transpose2d_input.Supports(
          input)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedInputArgumentError(
            input,
            context_properties.data_type_limits.conv_transpose2d_input)));
  }
  const auto input_info = GetConv2dInputInfo(label, input, attributes);

  // Validate filter operand.
  if (!context_properties.data_type_limits.conv_transpose2d_input.Supports(
          filter)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedArgumentError(
            kFilterParam, filter,
            context_properties.data_type_limits.conv_transpose2d_input)));
  }
  if (filter.data_type() != input.data_type()) {
    return base::unexpected(ErrorWithLabel(
        label, "The filter data type doesn't match the input data type."));
  }

  // Validate bias operand if it is present.
  if (attributes.bias_operand) {
    if (!context_properties.data_type_limits.conv_transpose2d_bias.Supports(
            attributes.bias_operand.value())) {
      return base::unexpected(ErrorWithLabel(
          label,
          NotSupportedArgumentError(
              kBiasParam, attributes.bias_operand.value(),
              context_properties.data_type_limits.conv_transpose2d_bias)));
    }
  }

  const std::vector<uint32_t>& filter_shape = filter.shape();
  uint32_t input_channels, filter_height, filter_width, filter_output_channels;
  // The conv2d filter layout specifies the filter layout format.
  switch (attributes.filter_layout) {
    case ConvTranspose2dFilterOperandLayout::kIohw:
      // "iohw": [input_channels, output_channels/groups, height, width]
      input_channels = filter_shape[0];
      filter_output_channels = filter_shape[1];
      filter_height = filter_shape[2];
      filter_width = filter_shape[3];
      break;
    case ConvTranspose2dFilterOperandLayout::kHwoi:
      // "hwoi": [height, width, output_channels/groups, input_channels]
      filter_height = filter_shape[0];
      filter_width = filter_shape[1];
      filter_output_channels = filter_shape[2];
      input_channels = filter_shape[3];
      break;
    case ConvTranspose2dFilterOperandLayout::kOhwi:
      // "ohwi": [output_channels/groups, height, width, input_channels]
      filter_output_channels = filter_shape[0];
      filter_height = filter_shape[1];
      filter_width = filter_shape[2];
      input_channels = filter_shape[3];
      break;
  }
  // Validate groups, input channels and calculate output channels.
  if (attributes.groups == 0) {
    return base::unexpected(
        ErrorWithLabel(label, "The groups should be greater than 0."));
  }
  if (input_info.channels != input_channels) {
    return base::unexpected(ErrorWithLabel(
        label, "The input channels should equal to filter input channels."));
  }
  const auto checked_output_channels =
      base::CheckedNumeric<uint32_t>(filter_output_channels) *
      attributes.groups;
  if (!checked_output_channels.IsValid()) {
    return base::unexpected(
        ErrorWithLabel(label, "The output channels is too large."));
  }
  const uint32_t output_channels = checked_output_channels.ValueOrDie();

  RETURN_IF_ERROR(ValidateIntermediatePaddedDescriptor(
      context_properties, input, attributes.padding, attributes.input_layout,
      input_info, label));

  // Validate and calculate output sizes.
  uint32_t output_height, output_width;
  if (attributes.output_sizes) {
    const auto& output_sizes = attributes.output_sizes;
    output_height = output_sizes->height;
    output_width = output_sizes->width;
    if (output_height <= 0 || output_width <= 0) {
      return base::unexpected(
          ErrorWithLabel(label, "All output sizes should be greater than 0."));
    }
    const auto strides = attributes.strides;
    ASSIGN_OR_RETURN(
        Size2d<uint32_t> calculated_output_sizes,
        ValidateAndCalculateConvTranspose2dOutputSizes(
            input_info.height, input_info.width, filter_height, filter_width,
            attributes.padding, strides, attributes.dilations,
            // According to WebNN spec:
            // https://webmachinelearning.github.io/webnn/#dom-mlconvtranspose2doptions-outputsizes
            // When the output sizes are explicitly specified, the output
            // padding values in outputPadding are ignored.
            {0, 0}, label));
    const auto calculated_output_height = calculated_output_sizes.height;
    const auto max_output_height =
        base::CheckedNumeric<uint32_t>(calculated_output_height) +
        strides.height;
    if (!max_output_height.IsValid()) {
      return base::unexpected(ErrorWithLabel(
          label, "The checked maximum output height is too large"));
    }
    if (output_height < calculated_output_height ||
        output_height >= max_output_height.ValueOrDie()) {
      return base::unexpected(
          ErrorWithLabel(label, "The height of output sizes is invalid."));
    }
    const auto calculated_output_width = calculated_output_sizes.width;
    const auto max_output_width =
        base::CheckedNumeric<uint32_t>(calculated_output_width) + strides.width;
    if (!max_output_width.IsValid()) {
      return base::unexpected(ErrorWithLabel(
          label, "The checked maximum output width is too large"));
    }
    if (output_width < calculated_output_width ||
        output_width >= max_output_width.ValueOrDie()) {
      return base::unexpected(
          ErrorWithLabel(label, "The width of output sizes is invalid."));
    }
  } else {
    ASSIGN_OR_RETURN(
        Size2d<uint32_t> output_sizes,
        ValidateAndCalculateConvTranspose2dOutputSizes(
            input_info.height, input_info.width, filter_height, filter_width,
            attributes.padding, attributes.strides, attributes.dilations,
            attributes.output_padding, label));
    output_height = output_sizes.height;
    output_width = output_sizes.width;
  }

  Conv2dInputOutputInfo output_info{.batches = input_info.batches,
                                    .channels = output_channels,
                                    .height = output_height,
                                    .width = output_width};
  return ValidateConv2dBiasAndCreateOutputOperand(context_properties, input,
                                                  attributes, output_info);
}

base::expected<OperandDescriptor, std::string>
ValidateCumulativeSumAndInferOutput(const ContextProperties& context_properties,
                                    const OperandDescriptor& input,
                                    const uint32_t axis,
                                    std::string_view label) {
  if (input.Rank() <= axis) {
    return base::unexpected(ErrorWithLabel(
        label, base::StringPrintf("The axis (%u) must be in the range [0, N-1] "
                                  "where N (%u) is the rank of input "
                                  "tensor.",
                                  axis, input.Rank())));
  }

  if (!context_properties.data_type_limits.cumulative_sum_input.Supports(
          input)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedInputArgumentError(
            input, context_properties.data_type_limits.cumulative_sum_input)));
  }

  // The data type and shape of input determine the output.
  return input;
}

// This helper method is intended to validate scale and zero_point
// operands of quantizeLinear and dequantizeLinear against the input
// operand.
// TODO(crbug.com/396176047): Make scale and zero_point's rank match with
// input.
base::expected<void, std::string>
ValidateScaleZeroPointOperandShapeIsCompatibleWithInput(
    base::span<const uint32_t> input_shape,
    base::span<const uint32_t> scale_shape,
    base::span<const uint32_t> zero_point_shape,
    std::string_view label) {
  // Check whether `scale_shape` is a subsample of `input_shape`.
  if (scale_shape.size() != input_shape.size()) {
    return base::unexpected(ErrorWithLabel(
        label, "The rank of scale is not equal to the rank of input."));
  }

  for (size_t i = 0; i < scale_shape.size(); ++i) {
    auto scale_dim = scale_shape[i];
    auto input_dim = input_shape[i];
    // The block_size should be an integer where block_size = dim_input /
    // dim_scale along the axis.
    if (input_dim % scale_dim != 0) {
      return base::unexpected(ErrorWithLabel(
          label,
          "The shape of scale is not a subsample of the shape of input."));
    }
  }

  if (!std::ranges::equal(scale_shape, zero_point_shape)) {
    return base::unexpected(ErrorWithLabel(
        label, "The shape of scale and zero point must be the same."));
  }
  return base::ok();
}

base::expected<OperandDescriptor, std::string>
ValidateDequantizeLinearAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    const OperandDescriptor& scale,
    const OperandDescriptor& zero_point,
    std::string_view label) {
  // Validate scale and zero_point operands.
  RETURN_IF_ERROR(ValidateScaleZeroPointOperandShapeIsCompatibleWithInput(
      input.shape(), scale.shape(), zero_point.shape(), label));

  if (!context_properties.data_type_limits.dequantize_linear_input.Supports(
          input)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedInputArgumentError(
            input,
            context_properties.data_type_limits.dequantize_linear_input)));
  }

  if (!context_properties.data_type_limits.dequantize_linear_input.Supports(
          zero_point)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedArgumentError(
            kZeroPointParam, zero_point,
            context_properties.data_type_limits.dequantize_linear_input)));
  }

  if (input.data_type() != zero_point.data_type()) {
    return base::unexpected(ErrorWithLabel(
        label, "The data type of input and zero point must be the same."));
  }

  if (!context_properties.data_type_limits.dequantize_linear_scale.Supports(
          scale)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedArgumentError(
            kScaleParam, scale,
            context_properties.data_type_limits.dequantize_linear_scale)));
  }

  // The data type of scale determines the output type.
  return OperandDescriptor::Create(context_properties, scale.data_type(),
                                   input.shape(), label);
}

base::expected<OperandDescriptor, std::string>
ValidateQuantizeLinearAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    const OperandDescriptor& scale,
    const OperandDescriptor& zero_point,
    std::string_view label) {
  // Validate scale and zero_point operands.
  RETURN_IF_ERROR(ValidateScaleZeroPointOperandShapeIsCompatibleWithInput(
      input.shape(), scale.shape(), zero_point.shape(), label));

  if (!context_properties.data_type_limits.quantize_linear_input.Supports(
          input)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedInputArgumentError(
            input, context_properties.data_type_limits.quantize_linear_input)));
  }

  if (!context_properties.data_type_limits.quantize_linear_input.Supports(
          scale)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kScaleParam, scale,
                   context_properties.data_type_limits.quantize_linear_input)));
  }

  if (input.data_type() != scale.data_type()) {
    return base::unexpected(ErrorWithLabel(
        label, "The data type of input and scale must be the same."));
  }

  if (!context_properties.data_type_limits.quantize_linear_zero_point.Supports(
          zero_point)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedArgumentError(
            kZeroPointParam, zero_point,
            context_properties.data_type_limits.quantize_linear_zero_point)));
  }
  // The data type of zero_point determines the output type.
  return OperandDescriptor::Create(context_properties, zero_point.data_type(),
                                   input.shape(), label);
}

base::expected<OperandDescriptor, std::string> ValidateExpandAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    base::span<const uint32_t> new_shape,
    std::string_view label) {
  // Validate input operand.
  if (!context_properties.data_type_limits.expand_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedInputArgumentError(
                   input, context_properties.data_type_limits.expand_input)));
  }

  uint32_t new_rank = base::checked_cast<uint32_t>(new_shape.size());
  if (!context_properties.data_type_limits.expand_input.ranks.Supports(
          new_rank)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedOpOutputRankError(
            new_rank, context_properties.data_type_limits.expand_input.ranks)));
  }

  std::optional<std::vector<uint32_t>> output_shape =
      BroadcastShapes(input.shape(), new_shape,
                      /*bidirectional=*/false);
  if (!output_shape) {
    return base::unexpected(ErrorWithLabel(
        label, "The input shape is not broadcastable to the new shape."));
  }
  CHECK(new_shape == *output_shape);

  return OperandDescriptor::Create(context_properties, input.data_type(),
                                   *output_shape, label);
}

base::expected<OperandDescriptor, std::string> ValidateGatherAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    const OperandDescriptor& indices,
    const uint32_t axis,
    std::string_view label) {
  // Validate input operand.
  if (!context_properties.data_type_limits.gather_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedInputArgumentError(
                   input, context_properties.data_type_limits.gather_input)));
  }

  if (input.Rank() <= axis) {
    return base::unexpected(ErrorWithLabel(
        label, base::StringPrintf("The axis (%u) must be in the range [0, N-1] "
                                  "where N=%u is the rank of input tensor.",
                                  axis, input.Rank())));
  }

  // Validate indices operand.
  if (!context_properties.data_type_limits.gather_indices.Supports(indices)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kIndicesParam, indices,
                   context_properties.data_type_limits.gather_indices)));
  }

  auto checked_output_rank =
      base::CheckedNumeric<uint32_t>(input.Rank()) - 1 + indices.Rank();
  if (!checked_output_rank.IsValid()) {
    return base::unexpected(
        ErrorWithLabel(label, "The output rank is too large."));
  }

  std::vector<uint32_t> output_shape;
  output_shape.reserve(checked_output_rank.ValueOrDie());
  for (uint32_t i = 0; i < input.Rank(); ++i) {
    if (i == axis) {
      std::ranges::copy(indices.shape(), std::back_inserter(output_shape));
    } else {
      output_shape.push_back(input.shape()[i]);
    }
  }

  return OperandDescriptor::Create(context_properties, input.data_type(),
                                   output_shape, label);
}

base::expected<OperandDescriptor, std::string>
ValidateGatherElementsAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    const OperandDescriptor& indices,
    const uint32_t axis,
    std::string_view label) {
  // Validate input operand.
  if (!context_properties.data_type_limits.gather_elements_input.Supports(
          input)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedInputArgumentError(
            input, context_properties.data_type_limits.gather_elements_input)));
  }

  if (input.Rank() <= axis) {
    return base::unexpected(ErrorWithLabel(
        label, base::StringPrintf("The axis (%u) must be in the range [0, N-1] "
                                  "where N=%u is the rank of input "
                                  "tensor.",
                                  axis, input.Rank())));
  }

  // Validate indices operand.
  if (!context_properties.data_type_limits.gather_elements_indices.Supports(
          indices)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedArgumentError(
            kIndicesParam, indices,
            context_properties.data_type_limits.gather_elements_indices)));
  }

  if (input.Rank() != indices.Rank()) {
    return base::unexpected(ErrorWithLabel(
        label,
        base::StringPrintf(
            "The input rank (%u) must be equal to the indices rank (%u).",
            input.Rank(), indices.Rank())));
  }

  for (uint32_t i = 0; i < input.Rank(); ++i) {
    if (i == axis) {
      continue;
    }
    if (input.shape()[i] != indices.shape()[i]) {
      return base::unexpected(
          ErrorWithLabel(label,
                         "Except on the axis dimension, the input and indices "
                         "tensor must have the same dimension size."));
    }
  }

  return OperandDescriptor::Create(context_properties, input.data_type(),
                                   indices.shape(), label);
}

base::expected<OperandDescriptor, std::string> ValidateGatherNDAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    const OperandDescriptor& indices,
    std::string_view label) {
  // Validate input operand.
  if (!context_properties.data_type_limits.gather_nd_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedInputArgumentError(
            input, context_properties.data_type_limits.gather_nd_input)));
  }

  // Validate indices operand.
  if (!context_properties.data_type_limits.gather_nd_indices.Supports(
          indices)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kIndicesParam, indices,
                   context_properties.data_type_limits.gather_nd_indices)));
  }

  uint32_t indices_last_dimension_size = indices.shape()[indices.Rank() - 1];
  if (indices_last_dimension_size > input.Rank()) {
    return base::unexpected(ErrorWithLabel(
        label, base::StringPrintf(
                   "The last dimension size of indices (%u) must be less than "
                   "or equal to the input rank (%u).",
                   indices_last_dimension_size, input.Rank())));
  }

  auto checked_output_rank = base::CheckedNumeric(indices.Rank()) - 1 +
                             input.Rank() - indices_last_dimension_size;
  if (!checked_output_rank.IsValid()) {
    return base::unexpected(
        ErrorWithLabel(label, "The output rank is too large."));
  }

  std::vector<uint32_t> output_shape;
  output_shape.reserve(checked_output_rank.ValueOrDie());
  base::span<const uint32_t> indices_shape = indices.shape();
  std::ranges::copy(indices_shape.first(indices_shape.size() - 1),
                    std::back_inserter(output_shape));
  base::span<const uint32_t> input_shape = input.shape();
  std::ranges::copy(input_shape.subspan(indices_last_dimension_size),
                    std::back_inserter(output_shape));

  return OperandDescriptor::Create(context_properties, input.data_type(),
                                   output_shape, label);
}

base::expected<OperandDescriptor, std::string> ValidateGemmAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& a,
    const OperandDescriptor& b,
    const GemmAttributes& attributes) {
  const std::string& label = attributes.label;
  // Validate a and b operand.
  if (!context_properties.data_type_limits.gemm_a.Supports(a)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedArgumentError(kGemmAParam, a,
                                  context_properties.data_type_limits.gemm_a)));
  }

  if (!context_properties.data_type_limits.gemm_a.Supports(b)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedArgumentError(kGemmBParam, b,
                                  context_properties.data_type_limits.gemm_a)));
  }

  if (a.data_type() != b.data_type()) {
    return base::unexpected(ErrorWithLabel(
        label, "The data types of first two inputs don't match."));
  }

  std::vector<uint32_t> shape_a = a.shape();
  if (attributes.a_transpose) {
    std::ranges::reverse(shape_a);
  }
  // The second input 2-D tensor with shape [K, N] if bTranspose is false, or
  // [N, K] if bTranspose is true.
  std::vector<uint32_t> shape_b = b.shape();
  if (attributes.b_transpose) {
    std::ranges::reverse(shape_b);
  }
  // The number of columns in the first matrix must be equal to the number of
  // rows in the second matrix.
  if (shape_a[1] != shape_b[0]) {
    return base::unexpected(ErrorWithLabel(
        label,
        base::StringPrintf(
            "The number of columns (%u) in the %sfirst matrix isn't equal to "
            "the number of rows (%u) in the %ssecond matrix.",
            shape_a[1], attributes.a_transpose ? "transposed " : "", shape_b[0],
            attributes.b_transpose ? "transposed " : "")));
  };
  // The output is 2-D tensor of shape [M, N].
  std::vector<uint32_t> output_shape = {shape_a[0], shape_b[1]};
  // The third input tensor c is either a scalar, or of the shape that is
  // unidirectionally broadcastable to the output shape [M, N].
  if (attributes.c_operand) {
    if (!context_properties.data_type_limits.gemm_c.Supports(
            attributes.c_operand.value())) {
      return base::unexpected(ErrorWithLabel(
          label, NotSupportedArgumentError(
                     kGemmCParam, attributes.c_operand.value(),
                     context_properties.data_type_limits.gemm_c)));
    }

    if (attributes.c_operand->data_type() != a.data_type()) {
      return base::unexpected(ErrorWithLabel(
          label,
          "The third input data type doesn't match other inputs' data type."));
    }

    if (!BroadcastShapes(attributes.c_operand->shape(), output_shape,
                         /*bidirectional=*/false)) {
      return base::unexpected(ErrorWithLabel(
          label,
          "The third input tensor isn't unidirectionally broadcastable to the "
          "output tensor."));
    }
  }

  return OperandDescriptor::Create(context_properties, a.data_type(),
                                   output_shape, label);
}

base::expected<std::vector<OperandDescriptor>, std::string>
ValidateGruAndInferOutput(const ContextProperties& context_properties,
                          const OperandDescriptor& input,
                          const OperandDescriptor& weight,
                          const OperandDescriptor& recurrent_weight,
                          uint32_t steps,
                          uint32_t hidden_size,
                          const GruAttributes& attributes) {
  const std::string& label = attributes.label;
  if (steps <= 0) {
    return base::unexpected(
        ErrorWithLabel(label, "The steps must be greater than 0."));
  }
  if (hidden_size <= 0) {
    return base::unexpected(
        ErrorWithLabel(label, "The hidden size must be greater than 0."));
  }

  // Validate the input operand.
  if (!context_properties.data_type_limits.gru_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedInputArgumentError(
                   input, context_properties.data_type_limits.gru_input)));
  }

  const std::vector<uint32_t>& input_dimensions = input.shape();
  if (input_dimensions[0] != steps) {
    return base::unexpected(ErrorWithLabel(
        label, "The input dimension[0] must be equal to the steps."));
  }
  const auto batch_size = input_dimensions[1];
  const auto input_size = input_dimensions[2];
  auto checked_three_times_hidden_size = base::CheckedNumeric(hidden_size) * 3;
  uint32_t three_times_hidden_size;
  if (!checked_three_times_hidden_size.AssignIfValid(
          &three_times_hidden_size)) {
    return base::unexpected(
        ErrorWithLabel(label, "The hidden size is too large."));
  }
  const uint32_t num_directions =
      attributes.direction == RecurrentNetworkDirection::kBoth ? 2 : 1;

  // Validate the weight operand.
  if (!context_properties.data_type_limits.gru_input.Supports(weight)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kWeightParam, weight,
                   context_properties.data_type_limits.gru_input)));
  }
  std::array<uint32_t, 3> expected_weight_shape = {
      num_directions, three_times_hidden_size, input_size};
  RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
      weight, kWeightParam, expected_weight_shape, input.data_type(), label));

  // Validate the recurrent weight operand.
  if (!context_properties.data_type_limits.gru_input.Supports(
          recurrent_weight)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kRecurrentWeightParam, recurrent_weight,
                   context_properties.data_type_limits.gru_input)));
  }
  std::array<uint32_t, 3> expected_recurrent_weight_shape = {
      num_directions, three_times_hidden_size, hidden_size};
  RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
      recurrent_weight, kRecurrentWeightParam, expected_recurrent_weight_shape,
      input.data_type(), label));

  // Validate the bias operand.
  if (attributes.bias) {
    if (!context_properties.data_type_limits.gru_bias.Supports(
            attributes.bias.value())) {
      return base::unexpected(ErrorWithLabel(
          label, NotSupportedArgumentError(
                     kBiasParam, attributes.bias.value(),
                     context_properties.data_type_limits.gru_bias)));
    }
    std::array<uint32_t, 2> expected_bias_shape = {num_directions,
                                                   three_times_hidden_size};
    RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
        *attributes.bias, kBiasParam, expected_bias_shape, input.data_type(),
        label));
  }

  // Validate the recurrent bias operand.
  if (attributes.recurrent_bias) {
    if (!context_properties.data_type_limits.gru_bias.Supports(
            attributes.recurrent_bias.value())) {
      return base::unexpected(ErrorWithLabel(
          label, NotSupportedArgumentError(
                     kRecurrentBiasParam, attributes.recurrent_bias.value(),
                     context_properties.data_type_limits.gru_bias)));
    }
    std::array<uint32_t, 2> expected_recurrent_bias_shape = {
        num_directions, three_times_hidden_size};
    RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
        *attributes.recurrent_bias, kRecurrentBiasParam,
        expected_recurrent_bias_shape, input.data_type(), label));
  }

  // Validate the initial hidden state operand.
  if (attributes.initial_hidden_state) {
    if (!context_properties.data_type_limits.gru_input.Supports(
            attributes.initial_hidden_state.value())) {
      return base::unexpected(ErrorWithLabel(
          label,
          NotSupportedArgumentError(
              kInitialHiddenStateParam, attributes.initial_hidden_state.value(),
              context_properties.data_type_limits.gru_input)));
    }
    std::array<uint32_t, 3> expected_initial_hidden_state_shape = {
        num_directions, batch_size, hidden_size};
    RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
        *attributes.initial_hidden_state, kInitialHiddenStateParam,
        expected_initial_hidden_state_shape, input.data_type(), label));
  }

  if (attributes.activation_count != 2) {
    return base::unexpected(
        ErrorWithLabel(label, "The number of activations must be 2."));
  }

  std::vector<OperandDescriptor> outputs;
  ASSIGN_OR_RETURN(
      OperandDescriptor output,
      OperandDescriptor::Create(
          context_properties, input.data_type(),
          std::array{num_directions, batch_size, hidden_size}, label));
  outputs.push_back(std::move(output));
  if (attributes.return_sequence) {
    ASSIGN_OR_RETURN(
        OperandDescriptor return_sequence_output,
        OperandDescriptor::Create(
            context_properties, input.data_type(),
            std::array{steps, num_directions, batch_size, hidden_size}, label));
    outputs.push_back(std::move(return_sequence_output));
  }

  return outputs;
}

base::expected<OperandDescriptor, std::string> ValidateGruCellAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    const OperandDescriptor& weight,
    const OperandDescriptor& recurrent_weight,
    const OperandDescriptor& hidden_state,
    uint32_t hidden_size,
    const GruCellAttributes& attributes) {
  const std::string& label = attributes.label;
  if (hidden_size <= 0) {
    return base::unexpected(
        ErrorWithLabel(label, "The hidden size must be greater than 0."));
  }

  // Validate the input operand.
  if (!context_properties.data_type_limits.gru_cell_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedInputArgumentError(
                   input, context_properties.data_type_limits.gru_cell_input)));
  }

  const uint32_t batch_size = input.shape()[0];
  const uint32_t input_size = input.shape()[1];
  auto checked_three_times_hidden_size = base::CheckedNumeric(hidden_size) * 3;
  uint32_t three_times_hidden_size;
  if (!checked_three_times_hidden_size.AssignIfValid(
          &three_times_hidden_size)) {
    return base::unexpected(
        ErrorWithLabel(label, "The hidden size is too large."));
  }

  // Validate the weight operand.
  if (!context_properties.data_type_limits.gru_cell_input.Supports(weight)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kWeightParam, weight,
                   context_properties.data_type_limits.gru_cell_input)));
  }
  std::array<uint32_t, 2> expected_weight_shape = {three_times_hidden_size,
                                                   input_size};
  RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
      weight, kWeightParam, expected_weight_shape, input.data_type(), label));

  // Validate the recurrent weight operand.
  if (!context_properties.data_type_limits.gru_cell_input.Supports(
          recurrent_weight)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kRecurrentWeightParam, recurrent_weight,
                   context_properties.data_type_limits.gru_cell_input)));
  }
  std::array<uint32_t, 2> expected_recurrent_weight_shape = {
      three_times_hidden_size, hidden_size};
  RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
      recurrent_weight, kRecurrentWeightParam, expected_recurrent_weight_shape,
      input.data_type(), label));

  // Validate the hidden state operand.
  if (!context_properties.data_type_limits.gru_cell_input.Supports(
          hidden_state)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kHiddenStateParam, hidden_state,
                   context_properties.data_type_limits.gru_cell_input)));
  }
  std::array<uint32_t, 2> expected_hidden_state_shape = {batch_size,
                                                         hidden_size};
  RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
      hidden_state, kHiddenStateParam, expected_hidden_state_shape,
      input.data_type(), label));

  // Validate the bias operand.
  if (attributes.bias) {
    if (!context_properties.data_type_limits.gru_cell_bias.Supports(
            attributes.bias.value())) {
      return base::unexpected(ErrorWithLabel(
          label, NotSupportedArgumentError(
                     kBiasParam, attributes.bias.value(),
                     context_properties.data_type_limits.gru_cell_bias)));
    }
    std::array<uint32_t, 1> expected_bias_shape = {three_times_hidden_size};
    RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
        *attributes.bias, kBiasParam, expected_bias_shape, input.data_type(),
        label));
  }

  // Validate the recurrent bias operand.
  if (attributes.recurrent_bias) {
    if (!context_properties.data_type_limits.gru_cell_bias.Supports(
            attributes.recurrent_bias.value())) {
      return base::unexpected(ErrorWithLabel(
          label, NotSupportedArgumentError(
                     kRecurrentBiasParam, attributes.recurrent_bias.value(),
                     context_properties.data_type_limits.gru_cell_bias)));
    }
    std::array<uint32_t, 1> expected_recurrent_bias_shape = {
        three_times_hidden_size};
    RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
        *attributes.recurrent_bias, kRecurrentBiasParam,
        expected_recurrent_bias_shape, input.data_type(), label));
  }

  if (attributes.activation_count != 2) {
    return base::unexpected(
        ErrorWithLabel(label, "The number of activations must be 2."));
  }

  std::array<uint32_t, 2> output_shape{batch_size, hidden_size};
  return OperandDescriptor::Create(context_properties, input.data_type(),
                                   output_shape, label);
}

base::expected<OperandDescriptor, std::string>
ValidateInstanceNormalizationAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    const InstanceNormalizationAttributes& attributes) {
  const std::string& label = attributes.label;
  // Validate the input operand.
  if (!context_properties.data_type_limits.instance_normalization_input
           .Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedInputArgumentError(
            input,
            context_properties.data_type_limits.instance_normalization_input)));
  }

  uint32_t axis;
  switch (attributes.layout) {
    case InputOperandLayout::kNchw:
      axis = 1;
      break;
    case InputOperandLayout::kNhwc:
      axis = 3;
      break;
  }

  // Validate scale operand.
  if (attributes.scale.has_value()) {
    if (!context_properties.data_type_limits.instance_normalization_scale
             .Supports(attributes.scale.value())) {
      return base::unexpected(ErrorWithLabel(
          label,
          NotSupportedArgumentError(kScaleParam, attributes.scale.value(),
                                    context_properties.data_type_limits
                                        .instance_normalization_scale)));
    }
    RETURN_IF_ERROR(ValidateNormalizationOperandIsCompatibleWithInput(
        attributes.scale.value(), input.data_type(), input.shape()[axis], label,
        kScaleParam));
  }

  // Validate the bias operand.
  if (attributes.bias.has_value()) {
    if (!context_properties.data_type_limits.instance_normalization_scale
             .Supports(attributes.bias.value())) {
      return base::unexpected(ErrorWithLabel(
          label, NotSupportedArgumentError(kBiasParam, attributes.bias.value(),
                                           context_properties.data_type_limits
                                               .instance_normalization_scale)));
    }
    RETURN_IF_ERROR(ValidateNormalizationOperandIsCompatibleWithInput(
        attributes.bias.value(), input.data_type(), input.shape()[axis], label,
        kBiasParam));
  }

  return input;
}

base::expected<OperandDescriptor, std::string>
ValidateLayerNormalizationAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    base::span<const uint32_t> axes,
    const LayerNormalizationAttributes& attributes) {
  const std::string& label = attributes.label;
  // Validate the input operand.
  if (!context_properties.data_type_limits.layer_normalization_input.Supports(
          input)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedInputArgumentError(
            input,
            context_properties.data_type_limits.layer_normalization_input)));
  }

  // Ensure that the axes are all less than the input rank and have no
  // duplication.
  RETURN_IF_ERROR(ValidateAxes(axes, input.Rank(), label));

  const std::vector<uint32_t>& input_dimensions = input.shape();

  // The dimensions for layerNormalization to reduce along.
  std::vector<uint32_t> reduction_dimensions;
  reduction_dimensions.reserve(axes.size());
  std::ranges::transform(
      axes, std::back_inserter(reduction_dimensions),
      [&input_dimensions](uint32_t axis) { return input_dimensions[axis]; });

  // Validate the scale operand.
  if (attributes.scale.has_value()) {
    if (attributes.scale->data_type() != input.data_type()) {
      return base::unexpected(ErrorWithLabel(
          label,
          "For scale operand: the data type doesn't match the input data "
          "type."));
    }
    if (attributes.scale->shape() != reduction_dimensions) {
      return base::unexpected(ErrorWithLabel(
          label,
          "For scale operand: the shape doesn't match the axis dimensions of "
          "the input."));
    }
  }

  // Validate the bias operand.
  if (attributes.bias.has_value()) {
    if (attributes.bias->data_type() != input.data_type()) {
      return base::unexpected(
          ErrorWithLabel(label,
                         "For bias operand: the data type doesn't match the "
                         "input data type."));
    }
    if (attributes.bias->shape() != reduction_dimensions) {
      return base::unexpected(ErrorWithLabel(
          label,
          "For bias operand: the shape doesn't match the axis dimensions of "
          "the input."));
    }
  }

  return input;
}

base::expected<std::vector<OperandDescriptor>, std::string>
ValidateLstmAndInferOutput(const ContextProperties& context_properties,
                           const OperandDescriptor& input,
                           const OperandDescriptor& weight,
                           const OperandDescriptor& recurrent_weight,
                           const uint32_t steps,
                           const uint32_t hidden_size,
                           const LstmAttributes& attributes) {
  const std::string& label = attributes.label;
  if (steps <= 0) {
    return base::unexpected(
        ErrorWithLabel(label, "The steps must be greater than 0."));
  }
  if (hidden_size <= 0) {
    return base::unexpected(
        ErrorWithLabel(label, "The hidden size must be greater than 0."));
  }

  uint32_t four_times_hidden_size;
  auto checked_four_times_hidden_size = base::CheckedNumeric(hidden_size) * 4;
  if (!checked_four_times_hidden_size.AssignIfValid(&four_times_hidden_size)) {
    return base::unexpected(
        ErrorWithLabel(label, "The hidden size is too large."));
  }

  // Validate the input operand.
  if (!context_properties.data_type_limits.lstm_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedInputArgumentError(
                   input, context_properties.data_type_limits.lstm_input)));
  }

  const auto& input_dimensions = input.shape();
  if (input_dimensions[0] != steps) {
    return base::unexpected(ErrorWithLabel(
        label, "The input dimensions[0] must be equal to the steps."));
  }

  const uint32_t batch_size = input_dimensions[1];
  const uint32_t input_size = input_dimensions[2];
  const uint32_t direction_count =
      attributes.direction == RecurrentNetworkDirection::kBoth ? 2 : 1;

  // Validate the weight operand.
  if (!context_properties.data_type_limits.lstm_input.Supports(weight)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kWeightParam, weight,
                   context_properties.data_type_limits.lstm_input)));
  }
  uint32_t expected_weight_shape[3] = {direction_count, four_times_hidden_size,
                                       input_size};
  RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
      weight, kWeightParam, expected_weight_shape, input.data_type(), label));

  // Validate the recurrent weight operand.
  if (!context_properties.data_type_limits.lstm_input.Supports(
          recurrent_weight)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kRecurrentWeightParam, recurrent_weight,
                   context_properties.data_type_limits.lstm_input)));
  }
  uint32_t expected_recurrent_weight_shape[3] = {
      direction_count, four_times_hidden_size, hidden_size};
  RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
      recurrent_weight, kRecurrentWeightParam, expected_recurrent_weight_shape,
      input.data_type(), label));

  // Validate the bias operand.
  if (attributes.bias) {
    if (!context_properties.data_type_limits.lstm_bias.Supports(
            attributes.bias.value())) {
      return base::unexpected(ErrorWithLabel(
          label, NotSupportedArgumentError(
                     kBiasParam, attributes.bias.value(),
                     context_properties.data_type_limits.lstm_bias)));
    }
    uint32_t expected_bias_shape[2] = {direction_count, four_times_hidden_size};
    RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
        attributes.bias.value(), kBiasParam, expected_bias_shape,
        input.data_type(), label));
  }

  // Validate the recurrent bias operand.
  if (attributes.recurrent_bias) {
    if (!context_properties.data_type_limits.lstm_bias.Supports(
            attributes.recurrent_bias.value())) {
      return base::unexpected(ErrorWithLabel(
          label, NotSupportedArgumentError(
                     kRecurrentBiasParam, attributes.recurrent_bias.value(),
                     context_properties.data_type_limits.lstm_bias)));
    }
    uint32_t expected_recurrent_bias_shape[2] = {direction_count,
                                                 four_times_hidden_size};
    RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
        attributes.recurrent_bias.value(), kRecurrentBiasParam,
        expected_recurrent_bias_shape, input.data_type(), label));
  }

  // Validate the peephole weight operand.
  if (attributes.peephole_weight) {
    if (!context_properties.data_type_limits.lstm_bias.Supports(
            attributes.peephole_weight.value())) {
      return base::unexpected(ErrorWithLabel(
          label, NotSupportedArgumentError(
                     kPeepholeWeightParam, attributes.peephole_weight.value(),
                     context_properties.data_type_limits.lstm_bias)));
    }
    // Here `3 * hidden_size` will not overflow because `4 * hidden_size` has
    // already been checked.
    uint32_t expected_peephole_weight_shape[2] = {direction_count,
                                                  3 * hidden_size};
    RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
        attributes.peephole_weight.value(), kPeepholeWeightParam,
        expected_peephole_weight_shape, input.data_type(), label));
  }

  // Validate the initial hidden state operand.
  if (attributes.initial_hidden_state) {
    if (!context_properties.data_type_limits.lstm_input.Supports(
            attributes.initial_hidden_state.value())) {
      return base::unexpected(ErrorWithLabel(
          label,
          NotSupportedArgumentError(
              kInitialHiddenStateParam, attributes.initial_hidden_state.value(),
              context_properties.data_type_limits.lstm_input)));
    }
    uint32_t expected_initial_hidden_state_shape[3] = {direction_count,
                                                       batch_size, hidden_size};
    RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
        attributes.initial_hidden_state.value(), kInitialHiddenStateParam,
        expected_initial_hidden_state_shape, input.data_type(), label));
  }

  // Validate the initial cell state operand.
  if (attributes.initial_cell_state) {
    if (!context_properties.data_type_limits.lstm_input.Supports(
            attributes.initial_cell_state.value())) {
      return base::unexpected(ErrorWithLabel(
          label,
          NotSupportedArgumentError(
              kInitialCellStateParam, attributes.initial_cell_state.value(),
              context_properties.data_type_limits.lstm_input)));
    }
    uint32_t expected_initial_cell_state_shape[3] = {direction_count,
                                                     batch_size, hidden_size};
    RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
        attributes.initial_cell_state.value(), kInitialCellStateParam,
        expected_initial_cell_state_shape, input.data_type(), label));
  }

  if (attributes.activation_count != 3) {
    return base::unexpected(ErrorWithLabel(
        label, "The activations should be a sequence of length 3."));
  }

  std::vector<OperandDescriptor> outputs;
  ASSIGN_OR_RETURN(
      OperandDescriptor output,
      OperandDescriptor::Create(
          context_properties, input.data_type(),
          std::array{direction_count, batch_size, hidden_size}, label));
  outputs.push_back(output);
  outputs.push_back(std::move(output));
  if (attributes.return_sequence) {
    ASSIGN_OR_RETURN(
        OperandDescriptor return_sequence_output,
        OperandDescriptor::Create(
            context_properties, input.data_type(),
            std::array{steps, direction_count, batch_size, hidden_size},
            label));
    outputs.push_back(std::move(return_sequence_output));
  }

  return outputs;
}

base::expected<std::vector<OperandDescriptor>, std::string>
ValidateLstmCellAndInferOutput(const ContextProperties& context_properties,
                               const OperandDescriptor& input,
                               const OperandDescriptor& weight,
                               const OperandDescriptor& recurrent_weight,
                               const OperandDescriptor& hidden_state,
                               const OperandDescriptor& cell_state,
                               const uint32_t hidden_size,
                               const LstmCellAttributes& attributes) {
  const std::string& label = attributes.label;
  if (hidden_size <= 0) {
    return base::unexpected(
        ErrorWithLabel(label, "The hidden size must be greater than 0."));
  }

  uint32_t four_times_hidden_size;
  auto checked_four_times_hidden_size = base::CheckedNumeric(hidden_size) * 4;
  if (!checked_four_times_hidden_size.AssignIfValid(&four_times_hidden_size)) {
    return base::unexpected(
        ErrorWithLabel(label, "The hidden size is too large."));
  }

  // Validate the input operand.
  if (!context_properties.data_type_limits.lstm_cell_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedInputArgumentError(
            input, context_properties.data_type_limits.lstm_cell_input)));
  }

  const uint32_t batch_size = input.shape()[0];
  const uint32_t input_size = input.shape()[1];

  // Validate the weight operand.
  if (!context_properties.data_type_limits.lstm_cell_input.Supports(weight)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kWeightParam, weight,
                   context_properties.data_type_limits.lstm_cell_input)));
  }
  std::array<uint32_t, 2> expected_weight_shape = {four_times_hidden_size,
                                                   input_size};
  RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
      weight, kWeightParam, expected_weight_shape, input.data_type(), label));

  // Validate the hidden state operand.
  if (!context_properties.data_type_limits.lstm_cell_input.Supports(
          hidden_state)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kHiddenStateParam, hidden_state,
                   context_properties.data_type_limits.lstm_cell_input)));
  }
  std::array<uint32_t, 2> expected_hidden_state_shape = {batch_size,
                                                         hidden_size};
  RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
      hidden_state, kHiddenStateParam, expected_hidden_state_shape,
      input.data_type(), label));

  // Validate the cell state operand.
  if (!context_properties.data_type_limits.lstm_cell_input.Supports(
          cell_state)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kCellStateParam, cell_state,
                   context_properties.data_type_limits.lstm_cell_input)));
  }
  std::array<uint32_t, 2> expected_cell_state_shape = {batch_size, hidden_size};
  RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(cell_state, kCellStateParam,
                                                  expected_cell_state_shape,
                                                  input.data_type(), label));

  // Validate the recurrent weight operand.
  if (!context_properties.data_type_limits.lstm_cell_input.Supports(
          recurrent_weight)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kRecurrentWeightParam, recurrent_weight,
                   context_properties.data_type_limits.lstm_cell_input)));
  }
  std::array<uint32_t, 2> expected_recurrent_weight_shape = {
      four_times_hidden_size, hidden_size};
  RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
      recurrent_weight, kRecurrentWeightParam, expected_recurrent_weight_shape,
      input.data_type(), label));

  // Validate the bias operand.
  if (attributes.bias) {
    if (!context_properties.data_type_limits.lstm_cell_bias.Supports(
            attributes.bias.value())) {
      return base::unexpected(ErrorWithLabel(
          label, NotSupportedArgumentError(
                     kBiasParam, attributes.bias.value(),
                     context_properties.data_type_limits.lstm_cell_bias)));
    }
    std::array<uint32_t, 1> expected_bias_shape = {four_times_hidden_size};
    RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
        attributes.bias.value(), kBiasParam, expected_bias_shape,
        input.data_type(), label));
  }

  // Validate the recurrent bias operand.
  if (attributes.recurrent_bias) {
    if (!context_properties.data_type_limits.lstm_cell_bias.Supports(
            attributes.recurrent_bias.value())) {
      return base::unexpected(ErrorWithLabel(
          label, NotSupportedArgumentError(
                     kRecurrentBiasParam, attributes.recurrent_bias.value(),
                     context_properties.data_type_limits.lstm_cell_bias)));
    }
    std::array<uint32_t, 1> expected_recurrent_bias_shape = {
        four_times_hidden_size};
    RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
        attributes.recurrent_bias.value(), kRecurrentBiasParam,
        expected_recurrent_bias_shape, input.data_type(), label));
  }

  // Validate the peephole weight operand.
  if (attributes.peephole_weight) {
    if (!context_properties.data_type_limits.lstm_cell_bias.Supports(
            attributes.peephole_weight.value())) {
      return base::unexpected(ErrorWithLabel(
          label, NotSupportedArgumentError(
                     kPeepholeWeightParam, attributes.peephole_weight.value(),
                     context_properties.data_type_limits.lstm_cell_bias)));
    }
    // Here `3 * hidden_size` will not overflow because `4 * hidden_size` has
    // already been checked.
    std::array<uint32_t, 1> expected_peephole_weight_shape = {3 * hidden_size};
    RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
        attributes.peephole_weight.value(), kPeepholeWeightParam,
        expected_peephole_weight_shape, input.data_type(), label));
  }

  if (attributes.activation_count != 3) {
    return base::unexpected(ErrorWithLabel(
        label, "The activations should be a sequence of length 3."));
  }

  std::vector<OperandDescriptor> outputs;
  outputs.reserve(2);

  ASSIGN_OR_RETURN(
      OperandDescriptor output,
      OperandDescriptor::Create(context_properties, input.data_type(),
                                std::array{batch_size, hidden_size}, label));
  outputs.push_back(output);
  outputs.push_back(std::move(output));

  return outputs;
}

base::expected<OperandDescriptor, std::string> ValidateMatmulAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& a,
    const OperandDescriptor& b,
    std::string_view label) {
  if (!context_properties.data_type_limits.matmul_input.Supports(a)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   "a", a, context_properties.data_type_limits.matmul_input)));
  }

  if (!context_properties.data_type_limits.matmul_input.Supports(b)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   "b", b, context_properties.data_type_limits.matmul_input)));
  }

  if (a.data_type() != b.data_type()) {
    return base::unexpected(ErrorWithLabel(
        label, "The data types of first two inputs don't match."));
  }

  std::vector<uint32_t> a_dimensions = a.shape();
  if (a_dimensions.size() < 2 || b.shape().size() < 2) {
    return base::unexpected(
        ErrorWithLabel(label, "MatMul inputs must be at least 2-D."));
  }
  std::vector<uint32_t> b_dimensions = b.shape();

  // The number of columns in the first matrix must be equal to the number of
  // rows in the second matrix.
  const uint32_t a_cols = a_dimensions[a_dimensions.size() - 1];
  const uint32_t a_rows = a_dimensions[a_dimensions.size() - 2];
  const uint32_t b_cols = b_dimensions[b_dimensions.size() - 1];
  const uint32_t b_rows = b_dimensions[b_dimensions.size() - 2];
  if (a_cols != b_rows) {
    return base::unexpected(ErrorWithLabel(
        label,
        base::StringPrintf(
            "The number of columns (%u) in the first matrix isn't equal to "
            "the number of rows (%u) in the second matrix.",
            a_cols, b_rows)));
  }

  size_t output_rank
