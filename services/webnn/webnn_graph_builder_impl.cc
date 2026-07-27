// Copyright 2024 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/webnn_graph_builder_impl.h"

#include <variant>

#include "base/check_is_test.h"
#include "base/containers/fixed_flat_map.h"
#include "base/containers/flat_map.h"
#include "base/feature_list.h"
#include "base/functional/callback_helpers.h"
#include "base/memory/raw_ref.h"
#include "base/memory/stack_allocated.h"
#include "base/numerics/checked_math.h"
#include "base/numerics/safe_conversions.h"
#include "base/task/bind_post_task.h"
#include "base/task/thread_pool.h"
#include "base/types/expected.h"
#include "base/types/fixed_array.h"
#include "base/types/pass_key.h"
#include "services/webnn/error.h"
#include "services/webnn/graph_builder_context.h"
#include "services/webnn/public/cpp/graph_validation_utils.h"
#include "services/webnn/public/cpp/operand_descriptor.h"
#include "services/webnn/public/cpp/supported_data_types.h"
#include "services/webnn/public/cpp/webnn_trace.h"
#include "services/webnn/public/cpp/webnn_types.h"
#include "services/webnn/public/mojom/webnn_error.mojom.h"
#include "services/webnn/webnn_constant_operand.h"
#include "services/webnn/webnn_graph_impl.h"
#include "services/webnn/webnn_pending_constant_operand.h"
#include "services/webnn/webnn_utils.h"
#include "third_party/tflite/buildflags.h"

#if BUILDFLAG(BUILD_TFLITE_WITH_XNNPACK)
#include "third_party/xnnpack/src/include/xnnpack.h"  // nogncheck
#endif  // BUILDFLAG(BUILD_TFLITE_WITH_XNNPACK)

// Evaluate `condition`, and if it returns false then return false.
#define RETURN_IF_FALSE(condition) \
  do {                             \
    if (!(condition))              \
      return false;                \
  } while (0)

namespace webnn {

namespace {

#if BUILDFLAG(BUILD_TFLITE_WITH_XNNPACK)
// Use XNNPACK to accelerate TransposePendingPermutation.
BASE_FEATURE(kWebNNUseXNNPackForConstantTransposeFolding,
             base::FEATURE_ENABLED_BY_DEFAULT);
#endif  // BUILDFLAG(BUILD_TFLITE_WITH_XNNPACK)

using DependentOperationsMap =
    base::flat_map<OperandId, base::flat_set<OperationId>>;

webnn::RecurrentNetworkDirection MojoRecurrentNetworkDirectionToComponent(
    mojom::RecurrentNetworkDirection direction) {
  switch (direction) {
    case mojom::RecurrentNetworkDirection::kForward:
      return webnn::RecurrentNetworkDirection::kForward;
    case mojom::RecurrentNetworkDirection::kBackward:
      return webnn::RecurrentNetworkDirection::kBackward;
    case mojom::RecurrentNetworkDirection::kBoth:
      return webnn::RecurrentNetworkDirection::kBoth;
  }
}

bool ValidateClampAttributes(const mojom::Clamp& clamp,
                             webnn::OperandDataType data_type) {
  if (clamp.min_value.IsNaN() || clamp.max_value.IsNaN()) {
    return false;
  }

  return !clamp.min_value.IsGreaterThan(clamp.max_value, data_type);
}

bool ValidateEluAttributes(const mojom::Elu& elu) {
  if (std::isnan(elu.alpha) || std::isinf(elu.alpha)) {
    // The value of alpha is nan.
    return false;
  }

  return true;
}

bool ValidateHardSigmoidAttributes(const mojom::HardSigmoid& hard_sigmoid) {
  if (std::isnan(hard_sigmoid.alpha) || std::isnan(hard_sigmoid.beta)) {
    // The value of alpha and beta should not be NAN.
    return false;
  }

  return true;
}

bool ValidateLeakyReluAttributes(const mojom::LeakyRelu& leaky_relu) {
  if (std::isnan(leaky_relu.alpha)) {
    // The value of alpha should not be NAN.
    return false;
  }

  return true;
}

bool ValidateLinearAttributes(const mojom::Linear& linear) {
  if (std::isnan(linear.alpha) || std::isnan(linear.beta)) {
    // The values of alpha and beta should not be NAN.
    return false;
  }

  return true;
}

const mojom::Operand* GetMojoOperand(
    base::span<const mojom::OperandPtr> operands,
    OperandId operand_id) {
  if (operand_id.value() >= operands.size()) {
    return nullptr;
  }
  return operands.at(operand_id.value()).get();
}

webnn::BatchNormalizationAttributes ConvertToBatchNormalizationAttributes(
    base::span<const mojom::OperandPtr> operands,
    const mojom::BatchNormalization& batch_normalization) {
  webnn::BatchNormalizationAttributes component_attributes;
  const auto& scale_operand_id = batch_normalization.scale_operand_id;
  if (scale_operand_id) {
    const mojom::Operand& scale_operand =
        *operands.at(*scale_operand_id.value());
    component_attributes.scale = scale_operand.descriptor;
  }
  const auto& bias_operand_id = batch_normalization.bias_operand_id;
  if (bias_operand_id) {
    const mojom::Operand& bias_operand = *operands.at(*bias_operand_id.value());
    component_attributes.bias = bias_operand.descriptor;
  }
  component_attributes.axis = batch_normalization.axis;
  component_attributes.label = batch_normalization.label;

  return component_attributes;
}

template <typename Conv2dAttributesType>
Conv2dAttributesType ConvertToConv2dAttributes(
    const webnn::ContextProperties& context_properties,
    base::span<const mojom::OperandPtr> operands,
    const webnn::mojom::Conv2d& conv2d,
    std::optional<OperandDescriptor> bias_operand) {
  Conv2dAttributesType attributes_base;
  // Convert padding, strides, dilations.
  auto& mojo_padding = conv2d.padding;
  attributes_base.padding = webnn::Padding2d{
      .beginning =
          webnn::Size2d<uint32_t>{.height = mojo_padding->beginning->height,
                                  .width = mojo_padding->beginning->width},
      .ending = webnn::Size2d<uint32_t>{.height = mojo_padding->ending->height,
                                        .width = mojo_padding->ending->width}};
  attributes_base.strides = webnn::Size2d<uint32_t>{
      .height = conv2d.strides->height, .width = conv2d.strides->width};
  attributes_base.dilations = webnn::Size2d<uint32_t>{
      .height = conv2d.dilations->height, .width = conv2d.dilations->width};

  // Convert groups, input layout and bias.
  attributes_base.groups = conv2d.groups;
  attributes_base.input_layout = context_properties.input_operand_layout;
  attributes_base.bias_operand = std::move(bias_operand);

  attributes_base.label = conv2d.label;

  return std::move(attributes_base);
}

webnn::Conv2dAttributes ConvertToConv2dAttributes(
    const webnn::ContextProperties& context_properties,
    base::span<const mojom::OperandPtr> operands,
    const webnn::mojom::Conv2d& conv2d,
    std::optional<OperandDescriptor> bias_operand) {
  auto component_attributes =
      ConvertToConv2dAttributes<webnn::Conv2dAttributes>(
          context_properties, operands, conv2d, std::move(bias_operand));
  switch (context_properties.input_operand_layout) {
    case webnn::InputOperandLayout::kNchw:
      // "channelsFirst": [batches, input_channels, height, width]
      component_attributes.filter_layout = Conv2dFilterOperandLayout::kOihw;
      break;
    case webnn::InputOperandLayout::kNhwc:
      // "channelsLast": [batches, height, width, input_channels]
      // For regular conv2d, ohwi filter layout is expected by default.
      // For depthwise conv2d, ihwo filter layout is expected by default.
      const auto* const input =
          GetMojoOperand(operands, conv2d.input_operand_id);
      CHECK(input);
      CHECK_EQ(input->descriptor.Rank(), 4u);
      const uint32_t input_channels = input->descriptor.shape()[3];
      const auto* const output =
          GetMojoOperand(operands, conv2d.output_operand_id);
      CHECK(output);
      CHECK_EQ(output->descriptor.Rank(), 4u);
      const uint32_t output_channels = output->descriptor.shape()[3];
      // Depthwise conv2d is "options.groups == input_channels ==
      // output_channels".
      const bool depthwise = webnn::IsDepthwiseConv2d(
          input_channels, output_channels, conv2d.groups);
      component_attributes.filter_layout =
          depthwise ? Conv2dFilterOperandLayout::kIhwo
                    : Conv2dFilterOperandLayout::kOhwi;
      break;
  }
  return component_attributes;
}

webnn::LstmAttributes ConvertToLstmAttributes(
    base::span<const mojom::OperandPtr> operands,
    const webnn::mojom::Lstm& lstm) {
  webnn::LstmAttributes attributes;
  attributes.return_sequence = lstm.return_sequence;
  attributes.direction =
      MojoRecurrentNetworkDirectionToComponent(lstm.direction);
  attributes.activation_count = lstm.activations.size();

  if (lstm.bias_operand_id.has_value()) {
    const auto* bias = GetMojoOperand(operands, lstm.bias_operand_id.value());
    attributes.bias = bias->descriptor;
  }
  if (lstm.recurrent_bias_operand_id.has_value()) {
    const auto* recurrent_bias =
        GetMojoOperand(operands, lstm.recurrent_bias_operand_id.value());
    attributes.recurrent_bias = recurrent_bias->descriptor;
  }
  if (lstm.peephole_weight_operand_id.has_value()) {
    const auto* peephole_weight =
        GetMojoOperand(operands, lstm.peephole_weight_operand_id.value());
    attributes.peephole_weight = peephole_weight->descriptor;
  }
  if (lstm.initial_hidden_state_operand_id.has_value()) {
    const auto* initial_hidden_state =
        GetMojoOperand(operands, lstm.initial_hidden_state_operand_id.value());
    attributes.initial_hidden_state = initial_hidden_state->descriptor;
  }
  if (lstm.initial_cell_state_operand_id.has_value()) {
    const auto* initial_cell_state =
        GetMojoOperand(operands, lstm.initial_cell_state_operand_id.value());
    attributes.initial_cell_state = initial_cell_state->descriptor;
  }
  attributes.label = lstm.label;

  return attributes;
}

webnn::LstmCellAttributes ConvertToLstmCellAttributes(
    base::span<const mojom::OperandPtr> operands,
    const webnn::mojom::LstmCell& lstm_cell) {
  webnn::LstmCellAttributes attributes;
  attributes.activation_count = lstm_cell.activations.size();

  if (lstm_cell.bias_operand_id.has_value()) {
    const auto* bias =
        GetMojoOperand(operands, lstm_cell.bias_operand_id.value());
    attributes.bias = bias->descriptor;
  }
  if (lstm_cell.recurrent_bias_operand_id.has_value()) {
    const auto* recurrent_bias =
        GetMojoOperand(operands, lstm_cell.recurrent_bias_operand_id.value());
    attributes.recurrent_bias = recurrent_bias->descriptor;
  }
  if (lstm_cell.peephole_weight_operand_id.has_value()) {
    const auto* peephole_weight =
        GetMojoOperand(operands, lstm_cell.peephole_weight_operand_id.value());
    attributes.peephole_weight = peephole_weight->descriptor;
  }
  attributes.label = lstm_cell.label;

  return attributes;
}

webnn::ConvTranspose2dAttributes ConvertToConvTranspose2dAttributes(
    const webnn::ContextProperties& context_properties,
    base::span<const mojom::OperandPtr> operands,
    const webnn::mojom::Conv2d& conv2d,
    std::optional<OperandDescriptor> bias_operand) {
  auto component_attributes =
      ConvertToConv2dAttributes<webnn::ConvTranspose2dAttributes>(
          context_properties, operands, conv2d, std::move(bias_operand));

  // Convert the output sizes that fetched from dimensions of output operand.
  auto* output = GetMojoOperand(operands, conv2d.output_operand_id);
  CHECK_EQ(output->descriptor.Rank(), 4u);
  webnn::Size2d<uint32_t> output_sizes;
  switch (context_properties.input_operand_layout) {
    case webnn::InputOperandLayout::kNchw:
      // "channelsFirst": [batches, input_channels, height, width]
      output_sizes.height = output->descriptor.shape()[2];
      output_sizes.width = output->descriptor.shape()[3];
      component_attributes.filter_layout =
          ConvTranspose2dFilterOperandLayout::kIohw;
      break;
    case webnn::InputOperandLayout::kNhwc:
      // "channelsLast": [batches, height, width, input_channels]
      output_sizes.height = output->descriptor.shape()[1];
      output_sizes.width = output->descriptor.shape()[2];
      component_attributes.filter_layout =
          ConvTranspose2dFilterOperandLayout::kOhwi;
      break;
  }
  component_attributes.output_sizes = std::move(output_sizes);

  return component_attributes;
}

webnn::LayerNormalizationAttributes ConvertToLayerNormalizationAttributes(
    base::span<const mojom::OperandPtr> operands,
    const mojom::LayerNormalization& layer_normalization) {
  webnn::LayerNormalizationAttributes component_attributes;
  const auto& scale_operand_id = layer_normalization.scale_operand_id;
  if (scale_operand_id.has_value()) {
    const mojom::Operand& scale_operand =
        *operands.at(*scale_operand_id.value());
    component_attributes.scale = scale_operand.descriptor;
  }

  const auto& bias_operand_id = layer_normalization.bias_operand_id;
  if (bias_operand_id.has_value()) {
    const mojom::Operand& bias_operand = *operands.at(*bias_operand_id.value());
    component_attributes.bias = bias_operand.descriptor;
  }
  component_attributes.label = layer_normalization.label;

  return component_attributes;
}

webnn::Pool2dAttributes ConvertToPool2dAttributes(
    const webnn::ContextProperties& context_properties,
    const webnn::mojom::Pool2d& pool2d,
    const mojom::Operand* output) {
  webnn::Pool2dAttributes component_attributes;
  auto& window_dimensions = pool2d.window_dimensions;
  component_attributes.window_dimensions = webnn::Size2d<uint32_t>{
      .height = window_dimensions->height, .width = window_dimensions->width};
  auto& mojo_padding = pool2d.padding;
  component_attributes.padding = webnn::Padding2d{
      .beginning =
          webnn::Size2d<uint32_t>{.height = mojo_padding->beginning->height,
                                  .width = mojo_padding->beginning->width},
      .ending = webnn::Size2d<uint32_t>{.height = mojo_padding->ending->height,
                                        .width = mojo_padding->ending->width}};
  component_attributes.strides = webnn::Size2d<uint32_t>{
      .height = pool2d.strides->height, .width = pool2d.strides->width};
  component_attributes.dilations = webnn::Size2d<uint32_t>{
      .height = pool2d.dilations->height, .width = pool2d.dilations->width};
  component_attributes.layout = context_properties.input_operand_layout;
  CHECK_EQ(output->descriptor.Rank(), 4u);
  switch (component_attributes.layout) {
    case webnn::InputOperandLayout::kNchw:
      component_attributes.output_sizes =
          webnn::Size2d<uint32_t>{.height = output->descriptor.shape()[2],
                                  .width = output->descriptor.shape()[3]};
      break;
    case webnn::InputOperandLayout::kNhwc:
      component_attributes.output_sizes =
          webnn::Size2d<uint32_t>{.height = output->descriptor.shape()[1],
                                  .width = output->descriptor.shape()[2]};
      break;
  }
  component_attributes.label = pool2d.label;

  return component_attributes;
}

webnn::GemmAttributes ConvertToGemmAttributes(
    base::span<const mojom::OperandPtr> operands,
    const mojom::Gemm& gemm) {
  webnn::GemmAttributes component_attributes;
  auto& c_operand_id = gemm.c_operand_id;
  if (c_operand_id) {
    const mojom::Operand& c_operand = *operands.at(*c_operand_id.value());
    component_attributes.c_operand = c_operand.descriptor;
  }
  component_attributes.alpha = gemm.alpha;
  component_attributes.beta = gemm.beta;
  component_attributes.a_transpose = gemm.a_transpose;
  component_attributes.b_transpose = gemm.b_transpose;
  component_attributes.label = gemm.label;

  return component_attributes;
}

webnn::GruAttributes ConvertToGruAttributes(
    base::span<const mojom::OperandPtr> operands,
    const webnn::mojom::Gru& gru) {
  webnn::GruAttributes component_attributes;
  if (gru.bias_operand_id.has_value()) {
    const auto* bias = GetMojoOperand(operands, gru.bias_operand_id.value());
    component_attributes.bias = bias->descriptor;
  }
  if (gru.recurrent_bias_operand_id.has_value()) {
    const auto* recurrent_bias =
        GetMojoOperand(operands, gru.recurrent_bias_operand_id.value());
    component_attributes.recurrent_bias = recurrent_bias->descriptor;
  }
  if (gru.initial_hidden_state_operand_id.has_value()) {
    const auto* initial_hidden_state =
        GetMojoOperand(operands, gru.initial_hidden_state_operand_id.value());
    component_attributes.initial_hidden_state =
        initial_hidden_state->descriptor;
  }

  component_attributes.return_sequence = gru.return_sequence;
  component_attributes.direction =
      MojoRecurrentNetworkDirectionToComponent(gru.direction);
  component_attributes.activation_count =
      base::checked_cast<uint32_t>(gru.activations.size());
  component_attributes.label = gru.label;

  return component_attributes;
}

webnn::GruCellAttributes ConvertToGruCellAttributes(
    base::span<const mojom::OperandPtr> operands,
    const webnn::mojom::GruCell& gru_cell) {
  webnn::GruCellAttributes component_attributes;
  if (gru_cell.bias_operand_id.has_value()) {
    const auto* bias =
        GetMojoOperand(operands, gru_cell.bias_operand_id.value());
    component_attributes.bias = bias->descriptor;
  }
  if (gru_cell.recurrent_bias_operand_id.has_value()) {
    const auto* recurrent_bias =
        GetMojoOperand(operands, gru_cell.recurrent_bias_operand_id.value());
    component_attributes.recurrent_bias = recurrent_bias->descriptor;
  }
  component_attributes.activation_count =
      base::checked_cast<uint32_t>(gru_cell.activations.size());
  component_attributes.label = gru_cell.label;

  return component_attributes;
}

webnn::InstanceNormalizationAttributes ConvertToInstanceNormalizationAttributes(
    const webnn::ContextProperties& context_properties,
    base::span<const mojom::OperandPtr> operands,
    const mojom::InstanceNormalization& instance_normalization) {
  webnn::InstanceNormalizationAttributes component_attributes;
  const auto& scale_operand_id = instance_normalization.scale_operand_id;
  if (scale_operand_id) {
    const mojom::Operand& scale_operand =
        *operands.at(*scale_operand_id.value());
    component_attributes.scale = scale_operand.descriptor;
  }
  const auto& bias_operand_id = instance_normalization.bias_operand_id;
  if (bias_operand_id) {
    const mojom::Operand& bias_operand = *operands.at(*bias_operand_id.value());
    component_attributes.bias = bias_operand.descriptor;
  }
  component_attributes.layout = context_properties.input_operand_layout;
  component_attributes.label = instance_normalization.label;

  return component_attributes;
}

webnn::SliceAttributes ConvertToSliceAttributes(
    const webnn::mojom::Slice& slice) {
  webnn::SliceAttributes component_attributes;
  component_attributes.starts.reserve(slice.ranges.size());
  component_attributes.sizes.reserve(slice.ranges.size());
  component_attributes.strides.reserve(slice.ranges.size());
  for (const auto& range : slice.ranges) {
    component_attributes.starts.push_back(range.start);
    component_attributes.sizes.push_back(range.size);
    component_attributes.strides.push_back(range.stride);
  }
  component_attributes.label = slice.label;

  return component_attributes;
}

std::vector<OperandId> GetOperationOutputs(const mojom::Operation& operation) {
  switch (operation.which()) {
    case mojom::Operation::Tag::kArgMinMax:
      return {operation.get_arg_min_max()->output_operand_id};
    case mojom::Operation::Tag::kBatchNormalization:
      return {operation.get_batch_normalization()->output_operand_id};
    case mojom::Operation::Tag::kClamp:
      return {operation.get_clamp()->output_operand_id};
    case mojom::Operation::Tag::kConcat:
      return {operation.get_concat()->output_operand_id};
    case mojom::Operation::Tag::kConv2d:
      return {operation.get_conv2d()->output_operand_id};
    case mojom::Operation::Tag::kCumulativeSum:
      return {operation.get_cumulative_sum()->output_operand_id};
    case mojom::Operation::Tag::kDequantizeLinear:
      return {operation.get_dequantize_linear()->output_operand_id};
    case mojom::Operation::Tag::kElementWiseBinary:
      return {operation.get_element_wise_binary()->output_operand_id};
    case mojom::Operation::Tag::kElu:
      return {operation.get_elu()->output_operand_id};
    case mojom::Operation::Tag::kElementWiseUnary:
      return {operation.get_element_wise_unary()->output_operand_id};
    case mojom::Operation::Tag::kExpand:
      return {operation.get_expand()->output_operand_id};
    case mojom::Operation::Tag::kGather:
      return {operation.get_gather()->output_operand_id};
    case mojom::Operation::Tag::kGatherElements:
      return {operation.get_gather_elements()->output_operand_id};
    case mojom::Operation::Tag::kGatherNd:
      return {operation.get_gather_nd()->output_operand_id};
    case mojom::Operation::Tag::kGelu:
      return {operation.get_gelu()->output_operand_id};
    case mojom::Operation::Tag::kGemm:
      return {operation.get_gemm()->output_operand_id};
    case mojom::Operation::Tag::kGru:
      return operation.get_gru()->output_operand_ids;
    case mojom::Operation::Tag::kGruCell:
      return {operation.get_gru_cell()->output_operand_id};
    case mojom::Operation::Tag::kHardSigmoid:
      return {operation.get_hard_sigmoid()->output_operand_id};
    case mojom::Operation::Tag::kHardSwish:
      return {operation.get_hard_swish()->output_operand_id};
    case mojom::Operation::Tag::kLayerNormalization:
      return {operation.get_layer_normalization()->output_operand_id};
    case mojom::Operation::Tag::kInstanceNormalization:
      return {operation.get_instance_normalization()->output_operand_id};
    case mojom::Operation::Tag::kLeakyRelu:
      return {operation.get_leaky_relu()->output_operand_id};
    case mojom::Operation::Tag::kLinear:
      return {operation.get_linear()->output_operand_id};
    case mojom::Operation::Tag::kLstm:
      return operation.get_lstm()->output_operand_ids;
    case mojom::Operation::Tag::kLstmCell:
      return operation.get_lstm_cell()->output_operand_ids;
    case mojom::Operation::Tag::kMatmul:
      return {operation.get_matmul()->output_operand_id};
    case mojom::Operation::Tag::kPad:
      return {operation.get_pad()->output_operand_id};
    case mojom::Operation::Tag::kPool2d:
      return {operation.get_pool2d()->output_operand_id};
    case mojom::Operation::Tag::kPrelu:
      return {operation.get_prelu()->output_operand_id};
    case mojom::Operation::Tag::kQuantizeLinear:
      return {operation.get_quantize_linear()->output_operand_id};
    case mojom::Operation::Tag::kReduce:
      return {operation.get_reduce()->output_operand_id};
    case mojom::Operation::Tag::kRelu:
      return {operation.get_relu()->output_operand_id};
    case mojom::Operation::Tag::kResample2d:
      return {operation.get_resample2d()->output_operand_id};
    case mojom::Operation::Tag::kReshape:
      return {operation.get_reshape()->output_operand_id};
    case mojom::Operation::Tag::kReverse:
      return {operation.get_reverse()->output_operand_id};
    case mojom::Operation::Tag::kScatterElements:
      return {operation.get_scatter_elements()->output_operand_id};
    case mojom::Operation::Tag::kScatterNd:
      return {operation.get_scatter_nd()->output_operand_id};
    case mojom::Operation::Tag::kSigmoid:
      return {operation.get_sigmoid()->output_operand_id};
    case mojom::Operation::Tag::kSlice:
      return {operation.get_slice()->output_operand_id};
    case mojom::Operation::Tag::kSoftmax:
      return {operation.get_softmax()->output_operand_id};
    case mojom::Operation::Tag::kSoftplus:
      return {operation.get_softplus()->output_operand_id};
    case mojom::Operation::Tag::kSoftsign:
      return {operation.get_softsign()->output_operand_id};
    case mojom::Operation::Tag::kSplit:
      return operation.get_split()->output_operand_ids;
    case mojom::Operation::Tag::kTanh:
      return {operation.get_tanh()->output_operand_id};
    case mojom::Operation::Tag::kTile:
      return {operation.get_tile()->output_operand_id};
    case mojom::Operation::Tag::kTranspose:
      return {operation.get_transpose()->output_operand_id};
    case mojom::Operation::Tag::kTriangular:
      return {operation.get_triangular()->output_operand_id};
    case mojom::Operation::Tag::kWhere:
      return {operation.get_where()->output_operand_id};
  }
}

// Helper class to validate a operations with the members passed to the
// constructor as context.
class OperationValidationContext {
  STACK_ALLOCATED();

 public:
  struct ValidationResult {
    base::flat_set<OperandId> processed_operands;
    DependentOperationsMap operand_to_dependent_operations;
    base::flat_map<OperandId, OperationId> operand_to_producing_operation;
  };

  // If `operations` are valid given the passed members as context, returns a
  // mapping of operands to the operations which depend on it.
  static std::optional<ValidationResult> ValidateOperationsAndGetDependencies(
      const std::vector<mojom::OperationPtr>& operations,
      const ContextProperties& context_properties,
      base::span<const mojom::OperandPtr> operands,
      base::flat_set<OperandId> processed_operands);

 private:
  OperationValidationContext(const ContextProperties& context_properties,
                             base::span<const mojom::OperandPtr> operands,
                             base::flat_set<OperandId> processed_operands)
      : context_properties_(context_properties),
        operands_(operands),
        processed_operands_(std::move(processed_operands)) {
    operand_to_dependent_operations_.reserve(operands.size());
    operand_to_producing_operation_.reserve(operands.size());
  }

  const mojom::Operand* GetMojoOperand(OperandId operand_id);

  void NoteInputDependency(OperandId operand_id, OperationId operation_id);
  bool NoteOutputDependency(const mojom::Operation& operation,
                            OperationId operation_id);

  bool IsProcessedOperand(OperandId operand_id);

  template <typename Operation>
  bool ValidateUnaryOperation(const Operation& operation,
                              const webnn::SupportedTensors& input_constraint,
                              OperationId operation_id);

  bool ValidateCastOperation(const mojom::ElementWiseUnary& operation,
                             OperationId operation_id);
  bool ValidateBatchNormalization(
      const mojom::BatchNormalization& batch_normalization,
      OperationId operation_id);
  bool ValidateArgMinMax(const mojom::ArgMinMax& arg_min_max,
                         OperationId operation_id);
  bool ValidateClamp(const mojom::Clamp& clamp, OperationId operation_id);
  bool ValidateConcat(const mojom::Concat& concat, OperationId operation_id);
  bool ValidateConv2d(const mojom::Conv2d& conv2d, OperationId operation_id);
  bool ValidateCumulativeSum(const mojom::CumulativeSum& cumulative_sum,
                             OperationId operation_id);
  bool ValidateDequantizeLinear(
      const mojom::DequantizeLinear& dequantize_linear,
      OperationId operation_id);
  bool ValidateElementWiseBinaryOperands(
      const mojom::Operand* lhs,
      const mojom::Operand* rhs,
      const mojom::Operand* output,
      const mojom::ElementWiseBinary& operation);
  bool ValidateElementWiseBinary(const mojom::ElementWiseBinary& operation,
                                 OperationId operation_id);
  bool ValidateElu(const mojom::Elu& elu, OperationId operation_id);

  bool ValidateElementWiseUnary(const mojom::ElementWiseUnary& operation,
                                OperationId operation_id);
  bool ValidateExpand(const mojom::Expand& expand, OperationId operation_id);
  bool ValidateGather(const mojom::Gather& gather, OperationId operation_id);
  bool ValidateGatherElements(const mojom::GatherElements& gather_elements,
                              OperationId operation_id);
  bool ValidateGatherND(const mojom::GatherND& gather_nd,
                        OperationId operation_id);
  bool ValidateGemm(const mojom::Gemm& gemm, OperationId operation_id);
  bool ValidateGru(const mojom::Gru& gru, OperationId operation_id);
  bool ValidateGruCell(const mojom::GruCell& gru_cell,
                       OperationId operation_id);
  bool ValidateHardSigmoid(const mojom::HardSigmoid& hard_sigmoid,
                           OperationId operation_id);
  bool ValidateLayerNormalization(
      const mojom::LayerNormalization& layer_normalization,
      OperationId operation_id);
  bool ValidateLeakyRelu(const mojom::LeakyRelu& leaky_relu,
                         OperationId operation_id);
  bool ValidateLinear(const mojom::Linear& linear, OperationId operation_id);
  bool ValidateLstm(const mojom::Lstm& lstm, OperationId operation_id);
  bool ValidateLstmCell(const mojom::LstmCell& lstm_cell,
                        OperationId operation_id);
  bool ValidateInstanceNormalization(
      const mojom::InstanceNormalization& instance_normalization,
      OperationId operation_id);
  bool ValidateMatmul(const mojom::Matmul& matmul, OperationId operation_id);
  bool ValidatePad(const mojom::Pad& pad, OperationId operation_id);
  bool ValidatePool2d(const mojom::Pool2d& pool2d, OperationId operation_id);
  bool ValidatePrelu(const mojom::Prelu& prelu, OperationId operation_id);
  bool ValidateQuantizeLinear(const mojom::QuantizeLinear& quantize_linear,
                              OperationId operation_id);
  bool ValidateResample2d(const mojom::Resample2d& resample2d,
                          OperationId operation_id);
  bool ValidateReshape(const mojom::Reshape& reshape, OperationId operation_id);
  bool ValidateReverseOperation(const mojom::Reverse& reverse,
                                OperationId operation_id);
  bool ValidateScatterElements(const mojom::ScatterElements& scatter_elements,
                               OperationId operation_id);
  bool ValidateScatterND(const mojom::ScatterND& scatter_nd,
                         OperationId operation_id);
  bool ValidateSlice(const mojom::Slice& slice, OperationId operation_id);
  bool ValidateSoftmax(const mojom::Softmax& softmax, OperationId operation_id);
  bool ValidateSplit(const mojom::Split& split, OperationId operation_id);
  bool ValidateTile(const mojom::Tile& tile, OperationId operation_id);
  bool ValidateTranspose(const mojom::Transpose& transpose,
                         OperationId operation_id);
  bool ValidateTriangular(const mojom::Triangular& triangular,
                          OperationId operation_id);
  bool ValidateWhere(const mojom::Where& where, OperationId operation_id);
  bool ValidateReduce(const mojom::Reduce& reduce, OperationId operation_id);

  bool ValidateOperation(const mojom::Operation& operation,
                         OperationId operation_id);

  const base::raw_ref<const ContextProperties> context_properties_;
  base::span<const mojom::OperandPtr> operands_;

  base::flat_set<OperandId> processed_operands_;

  DependentOperationsMap operand_to_dependent_operations_;
  base::flat_map<OperandId, OperationId> operand_to_producing_operation_;
};

const mojom::Operand* OperationValidationContext::GetMojoOperand(
    OperandId operand_id) {
  return ::webnn::GetMojoOperand(operands_, operand_id);
}

void OperationValidationContext::NoteInputDependency(OperandId operand_id,
                                                     OperationId operation_id) {
  auto it = operand_to_dependent_operations_.find(operand_id);
  if (it == operand_to_dependent_operations_.end()) {
    operand_to_dependent_operations_.emplace(operand_id,
                                             std::vector({operation_id}));
  } else {
    it->second.insert(operation_id);
  }
}

bool OperationValidationContext::NoteOutputDependency(
    const mojom::Operation& operation,
    OperationId operation_id) {
  for (OperandId output_operand_id : GetOperationOutputs(operation)) {
    RETURN_IF_FALSE(operand_to_producing_operation_
                        .try_emplace(output_operand_id, operation_id)
                        .second);
    RETURN_IF_FALSE(
        processed_operands_.insert(OperandId(output_operand_id)).second);
  }
  return true;
}

// static
std::optional<OperationValidationContext::ValidationResult>
OperationValidationContext::ValidateOperationsAndGetDependencies(
    const std::vector<mojom::OperationPtr>& operations,
    const ContextProperties& context_properties,
    base::span<const mojom::OperandPtr> operands,
    base::flat_set<OperandId> processed_operands) {
  OperationValidationContext context(context_properties, operands,
                                     std::move(processed_operands));

  for (size_t i = 0; i < operations.size(); i++) {
    if (!context.ValidateOperation(*operations[i], /*operation_id=*/i)) {
      return std::nullopt;
    }
  }

  return {{std::move(context.processed_operands_),
           std::move(context.operand_to_dependent_operations_),
           std::move(context.operand_to_producing_operation_)}};
}

bool OperationValidationContext::IsProcessedOperand(OperandId operand_id) {
  return operand_id.value() < operands_.size() &&
         processed_operands_.contains(operand_id);
}

template <typename Operation>
bool OperationValidationContext::ValidateUnaryOperation(
    const Operation& operation,
    const webnn::SupportedTensors& input_constraint,
    OperationId operation_id) {
  if (!IsProcessedOperand(operation.input_operand_id)) {
    return false;
  }
  NoteInputDependency(operation.input_operand_id, operation_id);

  const auto* input = GetMojoOperand(operation.input_operand_id);
  const auto* output = GetMojoOperand(operation.output_operand_id);
  if (!input || !output || output == input) {
    // The unary operator is invalid.
    return false;
  }

  if (!input_constraint.Supports(input->descriptor)) {
    // The data type is not in the constraint.
    return false;
  }

  if constexpr (std::is_same_v<Operation, mojom::ElementWiseUnary>) {
    if (IsLogicalElementWiseUnary(operation.kind)) {
      // For logical unary operations, output must be uint8 but shape should
      // match input.
      if (output->descriptor.data_type() != OperandDataType::kUint8) {
        return false;
      }
      return output->descriptor.shape() == input->descriptor.shape();
    }
  }

  // For all other operations, output descriptor should match input descriptor
  // exactly.
  return output->descriptor == input->descriptor;
}

bool OperationValidationContext::ValidateCastOperation(
    const mojom::ElementWiseUnary& operation,
    OperationId operation_id) {
  if (!IsProcessedOperand(operation.input_operand_id)) {
    return false;
  }
  NoteInputDependency(operation.input_operand_id, operation_id);

  const auto* input = GetMojoOperand(operation.input_operand_id);
  const auto* output = GetMojoOperand(operation.output_operand_id);
  if (!input || !output || output == input) {
    // The unary operator is invalid.
    return false;
  }

  const base::expected<OperandDescriptor, std::string> validated_output =
      ValidateCastAndInferOutput(*context_properties_, input->descriptor,
                                 output->descriptor.data_type(),
                                 operation.label);

  if (!validated_output.has_value()) {
    return false;
  }
  if (validated_output != output->descriptor) {
    return false;
  }

  return true;
}

bool OperationValidationContext::ValidateBatchNormalization(
    const mojom::BatchNormalization& batch_normalization,
    OperationId operation_id) {
  if (!IsProcessedOperand(batch_normalization.input_operand_id) ||
      !IsProcessedOperand(batch_normalization.mean_operand_id) ||
      !IsProcessedOperand(batch_normalization.variance_operand_id)) {
    return false;
  }
  NoteInputDependency(batch_normalization.input_operand_id, operation_id);
  NoteInputDependency(batch_normalization.mean_operand_id, operation_id);
  NoteInputDependency(batch_normalization.variance_operand_id, operation_id);

  const auto* input = GetMojoOperand(batch_normalization.input_operand_id);
  const auto* mean = GetMojoOperand(batch_normalization.mean_operand_id);
  const auto* variance =
      GetMojoOperand(batch_normalization.variance_operand_id);
  const auto* output = GetMojoOperand(batch_normalization.output_operand_id);
  if (!input || !mean || !variance || !output || output == input ||
      output == mean || output == variance) {
    // The batchNormalization operator is invalid.
    return false;
  }
  const auto& scale_operand_id = batch_normalization.scale_operand_id;
  if (scale_operand_id) {
    if (!IsProcessedOperand(scale_operand_id.value())) {
      // The scale operand is invalid.
      return false;
    }
    NoteInputDependency(scale_operand_id.value(), operation_id);

    auto* scale = GetMojoOperand(scale_operand_id.value());
    if (!scale || scale == output) {
      // The scale operand is invalid.
      return false;
    }
  }
  const auto& bias_operand_id = batch_normalization.bias_operand_id;
  if (bias_operand_id) {
    if (!IsProcessedOperand(bias_operand_id.value())) {
      // The bias operand is invalid.
      return false;
    }
    NoteInputDependency(bias_operand_id.value(), operation_id);

    auto* bias = GetMojoOperand(bias_operand_id.value());
    if (!bias || bias == output) {
      // The bias operand is invalid.
      return false;
    }
  }

  const base::expected<OperandDescriptor, std::string> validated_output =
      ValidateBatchNormalizationAndInferOutput(
          *context_properties_, input->descriptor, mean->descriptor,
          variance->descriptor,
          ConvertToBatchNormalizationAttributes(operands_,
                                                batch_normalization));
  if (!validated_output.has_value()) {
    return false;
  }
  if (validated_output != output->descriptor) {
    return false;
  }

  return true;
}

bool OperationValidationContext::ValidateArgMinMax(
    const mojom::ArgMinMax& arg_min_max,
    OperationId operation_id) {
  if (!IsProcessedOperand(arg_min_max.input_operand_id)) {
    return false;
  }
  NoteInputDependency(arg_min_max.input_operand_id, operation_id);

  const auto* input = GetMojoOperand(arg_min_max.input_operand_id);
  const auto* output = GetMojoOperand(arg_min_max.output_operand_id);
  if (!input || !output || output == input) {
    // The argMinMax operator is invalid.
    return false;
  }

  const base::expected<OperandDescriptor, std::string> validated_output =
      ValidateArgMinMaxAndInferOutput(*context_properties_, input->descriptor,
                                      arg_min_max.label, arg_min_max.axis,
                                      output->descriptor.data_type(),
                                      arg_min_max.keep_dimensions);
  if (!validated_output.has_value()) {
    return false;
  }
  if (validated_output != output->descriptor) {
    return false;
  }

  return true;
}

bool OperationValidationContext::ValidateClamp(const mojom::Clamp& clamp,
                                               OperationId operation_id) {
  if (!ValidateUnaryOperation(clamp,
                              context_properties_->data_type_limits.clamp_input,
                              operation_id)) {
    return false;
  }
  const auto* input = GetMojoOperand(clamp.input_operand_id);
  if (!ValidateClampAttributes(clamp, input->descriptor.data_type())) {
    return false;
  }

  return true;
}

bool OperationValidationContext::ValidateConcat(const mojom::Concat& concat,
                                                OperationId operation_id) {
  auto* output = GetMojoOperand(concat.output_operand_id);
  if (!output) {
    // The concat operator is invalid.
    return false;
  }

  std::vector<OperandDescriptor> inputs;
  inputs.reserve(concat.input_operand_ids.size());
  for (const auto& input_operand_id : concat.input_operand_ids) {
    if (!IsProcessedOperand(input_operand_id)) {
      return false;
    }
    NoteInputDependency(input_operand_id, operation_id);

    auto* input = GetMojoOperand(input_operand_id);
    if (!input || input == output) {
      return false;
    }
    inputs.push_back(input->descriptor);
  }

  const base::expected<OperandDescriptor, std::string> validated_output =
      ValidateConcatAndInferOutput(*context_properties_, inputs, concat.axis,
                                   concat.label);
  if (!validated_output.has_value()) {
    return false;
  }
  if (validated_output != output->descriptor) {
    return false;
  }

  return true;
}

bool OperationValidationContext::ValidateConv2d(const mojom::Conv2d& conv2d,
                                                OperationId operation_id) {
  if (!IsProcessedOperand(conv2d.input_operand_id) ||
      !IsProcessedOperand(conv2d.filter_operand_id)) {
    return false;
  }
  NoteInputDependency(conv2d.input_operand_id, operation_id);
  NoteInputDependency(conv2d.filter_operand_id, operation_id);

  auto* input = GetMojoOperand(conv2d.input_operand_id);
  auto* filter = GetMojoOperand(conv2d.filter_operand_id);
  auto* output = GetMojoOperand(conv2d.output_operand_id);
  if (!input || !filter || !output || output == input || output == filter) {
    // The conv2d operator is invalid.
    return false;
  }

  // The input, filter, and output rank need to be validated before converting
  // to `webnn::Conv2dAttributes`.
  if (input->descriptor.Rank() != 4 || output->descriptor.Rank() != 4 ||
      filter->descriptor.Rank() != 4) {
    // The element of input, filter, and output dimensions should be 4.
    return false;
  }

  std::optional<OperandDescriptor> bias_operand;
  auto& bias_operand_id = conv2d.bias_operand_id;
  if (bias_operand_id) {
    if (!IsProcessedOperand(bias_operand_id.value())) {
      return false;
    }
    NoteInputDependency(bias_operand_id.value(), operation_id);

    auto* bias = GetMojoOperand(bias_operand_id.value());
    if (!bias || bias == output) {
      // Invalid bias operand.
      return false;
    }
    bias_operand = bias->descriptor;
  }

  std::optional<base::expected<OperandDescriptor, std::string>>
      validated_output;
  switch (conv2d.kind) {
    case mojom::Conv2d::Kind::kDirect: {
      validated_output = ValidateConv2dAndInferOutput(
          *context_properties_, input->descriptor, filter->descriptor,
          ConvertToConv2dAttributes(*context_properties_, operands_, conv2d,
                                    std::move(bias_operand)));
      break;
    }

    case mojom::Conv2d::Kind::kTransposed: {
      validated_output = ValidateConvTranspose2dAndInferOutput(
          *context_properties_, input->descriptor, filter->descriptor,
          ConvertToConvTranspose2dAttributes(*context_properties_, operands_,
                                             conv2d, std::move(bias_operand)));
      break;
    }
  }
  if (!validated_output.has_value()) {
    return false;
  }
  if (validated_output != output->descriptor) {
    return false;
  }

  return true;
}

bool OperationValidationContext::ValidateCumulativeSum(
    const mojom::CumulativeSum& cumulative_sum,
    OperationId operation_id) {
  if (!IsProcessedOperand(cumulative_sum.input_operand_id)) {
    return false;
  }
  NoteInputDependency(cumulative_sum.input_operand_id, operation_id);

  auto* input = GetMojoOperand(cumulative_sum.input_operand_id);
  auto* output = GetMojoOperand(cumulative_sum.output_operand_id);

  if (!input || !output || output == input) {
    // The cumulative_sum operator is invalid.
    return false;
  }

  const base::expected<OperandDescriptor, std::string> validated_output =
      ValidateCumulativeSumAndInferOutput(
          *context_properties_, input->descriptor, cumulative_sum.axis,
          cumulative_sum.label);
  if (!validated_output.has_value()) {
    return false;
  }
  if (validated_output != output->descriptor) {
    return false;
  }

  return true;
}

bool OperationValidationContext::ValidateDequantizeLinear(
    const mojom::DequantizeLinear& dequantize_linear,
    OperationId operation_id) {
  if (!IsProcessedOperand(dequantize_linear.input_operand_id) ||
      !IsProcessedOperand(dequantize_linear.scale_operand_id) ||
      !IsProcessedOperand(dequantize_linear.zero_point_operand_id)) {
    return false;
  }
  NoteInputDependency(dequantize_linear.input_operand_id, operation_id);
  NoteInputDependency(dequantize_linear.scale_operand_id, operation_id);
  NoteInputDependency(dequantize_linear.zero_point_operand_id, operation_id);

  auto* input = GetMojoOperand(dequantize_linear.input_operand_id);
  auto* output = GetMojoOperand(dequantize_linear.output_operand_id);
  auto* scale = GetMojoOperand(dequantize_linear.scale_operand_id);
  auto* zero_point = GetMojoOperand(dequantize_linear.zero_point_operand_id);
  if (!input || !output || !scale || !zero_point || output == input ||
      output == scale || output == zero_point) {
    // The quantize_linear operator is invalid.
    return false;
  }

  const base::expected<OperandDescriptor, std::string> validated_output =
      ValidateDequantizeLinearAndInferOutput(
          *context_properties_, input->descriptor, scale->descriptor,
          zero_point->descriptor, dequantize_linear.label);
  if (!validated_output.has_value()) {
    return false;
  }
  if (validated_output != output->descriptor) {
    return false;
  }

  return true;
}

bool OperationValidationContext::ValidateElementWiseBinaryOperands(
    const mojom::Operand* lhs,
    const mojom::Operand* rhs,
    const mojom::Operand* output,
    const mojom::ElementWiseBinary& operation) {
  if (lhs->descriptor.data_type() != rhs->descriptor.data_type()) {
    // The input types don't match.
    return false;
  }

  if (IsLogicalElementWiseBinary(operation.kind)) {
    if (output->descriptor.data_type() != OperandDataType::kUint8) {
      // For logical operations, the output data type must be uint8.
      return false;
    }
  } else {
    // For all other operations, the input and output data types must match.
    if (output->descriptor.data_type() != lhs->descriptor.data_type()) {
      return false;
    }
  }

  switch (operation.kind) {
    case mojom::ElementWiseBinary::Kind::kAdd:
      return context_properties_->data_type_limits.add_input.SupportsAll(
          {lhs->descriptor, rhs->descriptor});
    case mojom::ElementWiseBinary::Kind::kSub:
      return context_properties_->data_type_limits.sub_input.SupportsAll(
          {lhs->descriptor, rhs->descriptor});
    case mojom::ElementWiseBinary::Kind::kMul:
      return context_properties_->data_type_limits.mul_input.SupportsAll(
          {lhs->descriptor, rhs->descriptor});
    case mojom::ElementWiseBinary::Kind::kDiv:
      return context_properties_->data_type_limits.div_input.SupportsAll(
          {lhs->descriptor, rhs->descriptor});
    case mojom::ElementWiseBinary::Kind::kMax:
      return context_properties_->data_type_limits.max_input.SupportsAll(
          {lhs->descriptor, rhs->descriptor});
    case mojom::ElementWiseBinary::Kind::kMin:
      return context_properties_->data_type_limits.min_input.SupportsAll(
          {lhs->descriptor, rhs->descriptor});
    case mojom::ElementWiseBinary::Kind::kPow:
      return context_properties_->data_type_limits.pow_input.SupportsAll(
          {lhs->descriptor, rhs->descriptor});
    case mojom::ElementWiseBinary::Kind::kEqual:
      return context_properties_->data_type_limits.equal_input.SupportsAll(
          {lhs->descriptor, rhs->descriptor});
    case mojom::ElementWiseBinary::Kind::kGreater:
      return context_properties_->data_type_limits.greater_input.SupportsAll(
          {lhs->descriptor, rhs->descriptor});
    case mojom::ElementWiseBinary::Kind::kGreaterOrEqual:
      return context_properties_->data_type_limits.greater_or_equal_input
          .SupportsAll({lhs->descriptor, rhs->descriptor});
    case mojom::ElementWiseBinary::Kind::kLesser:
      return context_properties_->data_type_limits.lesser_input.SupportsAll(
          {lhs->descriptor, rhs->descriptor});
    case mojom::ElementWiseBinary::Kind::kLesserOrEqual:
      return context_properties_->data_type_limits.lesser_or_equal_input
          .SupportsAll({lhs->descriptor, rhs->descriptor});
    case mojom::ElementWiseBinary::Kind::kNotEqual:
      return context_properties_->data_type_limits.not_equal_input.SupportsAll(
          {lhs->descriptor, rhs->descriptor});
    case mojom::ElementWiseBinary::Kind::kLogicalAnd:
      return context_properties_->data_type_limits.logical_and_input
          .SupportsAll({lhs->descriptor, rhs->descriptor});
    case mojom::ElementWiseBinary::Kind::kLogicalOr:
      return context_properties_->data_type_limits.logical_or_input.SupportsAll(
          {lhs->descriptor, rhs->descriptor});
    case mojom::ElementWiseBinary::Kind::kLogicalXor:
      return context_properties_->data_type_limits.logical_xor_input
          .SupportsAll({lhs->descriptor, rhs->descriptor});
  }
}

bool OperationValidationContext::ValidateElementWiseBinary(
    const mojom::ElementWiseBinary& operation,
    OperationId operation_id) {
  if (!IsProcessedOperand(operation.lhs_operand_id) ||
      !IsProcessedOperand(operation.rhs_operand_id)) {
    return false;
  }
  NoteInputDependency(operation.lhs_operand_id, operation_id);
  NoteInputDependency(operation.rhs_operand_id, operation_id);

  auto* a = GetMojoOperand(operation.lhs_operand_id);
  auto* b = GetMojoOperand(operation.rhs_operand_id);
  auto* output = GetMojoOperand(operation.output_operand_id);

  if (!a || !b || !output || output == a || output == b) {
    // The elementWise binary operator is invalid.
    return false;
  }

  if (!ValidateElementWiseBinaryOperands(a, b, output, operation)) {
    return false;
  }

  auto dims_output =
      BroadcastShapes(a->descriptor.shape(), b->descriptor.shape());
  if (!dims_output) {
    // The input shapes are not broadcastable.
    return false;
  }
  if (!std::ranges::equal(output->descriptor.shape(), dims_output.value())) {
    // The output shape is not expected.
    return false;
  }
  return true;
}

bool OperationValidationContext::ValidateElu(const mojom::Elu& elu,
                                             OperationId operation_id) {
  if (!ValidateUnaryOperation(
          elu, context_properties_->data_type_limits.elu_input, operation_id)) {
    return false;
  }

  if (!ValidateEluAttributes(elu)) {
    return false;
  }

  return true;
}

bool OperationValidationContext::ValidateElementWiseUnary(
    const mojom::ElementWiseUnary& operation,
    OperationId operation_id) {
  switch (operation.kind) {
    case mojom::ElementWiseUnary::Kind::kAbs:
      return ValidateUnaryOperation(
          operation, context_properties_->data_type_limits.abs_input,
          operation_id);
    case mojom::ElementWiseUnary::Kind::kCast:
      return ValidateCastOperation(operation, operation_id);
    case mojom::ElementWiseUnary::Kind::kCeil:
      return ValidateUnaryOperation(
          operation, context_properties_->data_type_limits.ceil_input,
          operation_id);
    case mojom::ElementWiseUnary::Kind::kCos:
      return ValidateUnaryOperation(
          operation, context_properties_->data_type_limits.cos_input,
          operation_id);
    case mojom::ElementWiseUnary::Kind::kErf:
      return ValidateUnaryOperation(
          operation, context_properties_->data_type_limits.erf_input,
          operation_id);
    case mojom::ElementWiseUnary::Kind::kExp:
      return ValidateUnaryOperation(
          operation, context_properties_->data_type_limits.exp_input,
          operation_id);
    case mojom::ElementWiseUnary::Kind::kFloor:
      return ValidateUnaryOperation(
          operation, context_properties_->data_type_limits.floor_input,
          operation_id);
    case mojom::ElementWiseUnary::Kind::kIdentity:
      return ValidateUnaryOperation(
          operation, context_properties_->data_type_limits.identity_input,
          operation_id);
    case mojom::ElementWiseUnary::Kind::kLog:
      return ValidateUnaryOperation(
          operation, context_properties_->data_type_limits.log_input,
          operation_id);
    case mojom::ElementWiseUnary::Kind::kIsNaN:
      return ValidateUnaryOperation(
          operation, context_properties_->data_type_limits.is_nan_input,
          operation_id);
    case mojom::ElementWiseUnary::Kind::kIsInfinite:
      return ValidateUnaryOperation(
          operation, context_properties_->data_type_limits.is_infinite_input,
          operation_id);
    case mojom::ElementWiseUnary::Kind::kLogicalNot:
      return ValidateUnaryOperation(
          operation, context_properties_->data_type_limits.logical_not_input,
          operation_id);
    case mojom::ElementWiseUnary::Kind::kNeg:
      return ValidateUnaryOperation(
          operation, context_properties_->data_type_limits.neg_input,
          operation_id);
    case mojom::ElementWiseUnary::Kind::kReciprocal:
      return ValidateUnaryOperation(
          operation, context_properties_->data_type_limits.reciprocal_input,
          operation_id);
    case mojom::ElementWiseUnary::Kind::kRoundEven:
      return ValidateUnaryOperation(
          operation, context_properties_->data_type_limits.round_even_input,
          operation_id);
    case mojom::ElementWiseUnary::Kind::kSign:
      return ValidateUnaryOperation(
          operation, context_properties_->data_type_limits.sign_input,
          operation_id);
    case mojom::ElementWiseUnary::Kind::kSin:
      return ValidateUnaryOperation(
          operation, context_properties_->data_type_limits.sin_input,
          operation_id);
    case mojom::ElementWiseUnary::Kind::kSqrt:
      return ValidateUnaryOperation(
          operation, context_properties_->data_type_limits.sqrt_input,
          operation_id);
    case mojom::ElementWiseUnary::Kind::kTan:
      return ValidateUnaryOperation(
          operation, context_properties_->data_type_limits.tan_input,
          operation_id);
  }
}

bool OperationValidationContext::ValidateExpand(const mojom::Expand& expand,
                                                OperationId operation_id) {
  if (!IsProcessedOperand(expand.input_operand_id)) {
    return false;
  }
  NoteInputDependency(expand.input_operand_id, operation_id);

  auto* input = GetMojoOperand(expand.input_operand_id);
  auto* output = GetMojoOperand(expand.output_operand_id);
  if (!input || !output || output == input) {
    // The expand operator is invalid.
    return false;
  }

  const base::expected<OperandDescriptor, std::string> validated_output =
      ValidateExpandAndInferOutput(*context_properties_, input->descriptor,
                                   output->descriptor.shape(), expand.label);
  if (!validated_output.has_value()) {
    return false;
  }
  if (validated_output != output->descriptor) {
    return false;
  }

  return true;
}

bool OperationValidationContext::ValidateGather(const mojom::Gather& gather,
                                                OperationId operation_id) {
  if (!IsProcessedOperand(gather.input_operand_id) ||
      !IsProcessedOperand(gather.indices_operand_id)) {
    return false;
  }
  NoteInputDependency(gather.input_operand_id, operation_id);
  NoteInputDependency(gather.indices_operand_id, operation_id);

  auto* input = GetMojoOperand(gather.input_operand_id);
  auto* output = GetMojoOperand(gather.output_operand_id);
  auto* indices = GetMojoOperand(gather.indices_operand_id);
  if (!input || !output || !indices || output == input || output == indices) {
    // The gather operator is invalid.
    return false;
  }

  const base::expected<OperandDescriptor, std::string> validated_output =
      ValidateGatherAndInferOutput(*context_properties_, input->descriptor,
                                   indices->descriptor, gather.axis,
                                   gather.label);
  if (!validated_output.has_value()) {
    return false;
  }
  if (validated_output != output->descriptor) {
    return false;
  }

  return true;
}

bool OperationValidationContext::ValidateGatherElements(
    const mojom::GatherElements& gather_elements,
    OperationId operation_id) {
  if (!IsProcessedOperand(gather_elements.input_operand_id) ||
      !IsProcessedOperand(gather_elements.indices_operand_id)) {
    return false;
  }
  NoteInputDependency(gather_elements.input_operand_id, operation_id);
  NoteInputDependency(gather_elements.indices_operand_id, operation_id);

  auto* input = GetMojoOperand(gather_elements.input_operand_id);
  auto* output = GetMojoOperand(gather_elements.output_operand_id);
  auto* indices = GetMojoOperand(gather_elements.indices_operand_id);
  if (!input || !output || !indices || output == input || output == indices) {
    return false;
  }

  const base::expected<OperandDescriptor, std::string> validated_output =
      ValidateGatherElementsAndInferOutput(
          *context_properties_, input->descriptor, indices->descriptor,
          gather_elements.axis, gather_elements.label);
  if (!validated_output.has_value()) {
    return false;
  }
  if (validated_output != output->descriptor) {
    return false;
  }

  return true;
}

bool OperationValidationContext::ValidateGatherND(
    const mojom::GatherND& gather_nd,
    OperationId operation_id) {
  if (!IsProcessedOperand(gather_nd.input_operand_id) ||
      !IsProcessedOperand(gather_nd.indices_operand_id)) {
    return false;
  }
  NoteInputDependency(gather_nd.input_operand_id, operation_id);
  NoteInputDependency(gather_nd.indices_operand_id, operation_id);

  auto* input = GetMojoOperand(gather_nd.input_operand_id);
  auto* output = GetMojoOperand(gather_nd.output_operand_id);
  auto* indices = GetMojoOperand(gather_nd.indices_operand_id);
  if (!input || !output || !indices || output == input || output == indices) {
    return false;
  }

  const base::expected<OperandDescriptor, std::string> validated_output =
      ValidateGatherNDAndInferOutput(*context_properties_, input->descriptor,
                                     indices->descriptor, gather_nd.label);
  if (!validated_output.has_value()) {
    return false;
  }
  if (validated_output != output->descriptor) {
    return false;
  }

  return true;
}

bool OperationValidationContext::ValidateGemm(const mojom::Gemm& gemm,
                                              OperationId operation_id) {
  if (!IsProcessedOperand(gemm.a_operand_id) ||
      !IsProcessedOperand(gemm.b_operand_id)) {
    return false;
  }
  NoteInputDependency(gemm.a_operand_id, operation_id);
  NoteInputDependency(gemm.b_operand_id, operation_id);

  auto* a = GetMojoOperand(gemm.a_operand_id);
  auto* b = GetMojoOperand(gemm.b_operand_id);
  auto* output = GetMojoOperand(gemm.output_operand_id);
  if (!a || !b || !output || output == a || output == b) {
    // The gemm operator is invalid.
    return false;
  }
  auto& c_operand_id = gemm.c_operand_id;
  if (c_operand_id) {
    if (!IsProcessedOperand(c_operand_id.value())) {
      // The third operand is invalid.
      return false;
    }
    NoteInputDependency(c_operand_id.value(), operation_id);

    auto* c = GetMojoOperand(c_operand_id.value());
    if (!c || c == output) {
      // The third operand is invalid.
      return false;
    }
  }

  const base::expected<OperandDescriptor, std::string> validated_output =
      ValidateGemmAndInferOutput(*context_properties_, a->descriptor,
                                 b->descriptor,
                                 ConvertToGemmAttributes(operands_, gemm));
  if (!validated_output.has_value()) {
    return false;
  }
  if (validated_output != output->descriptor) {
    return false;
  }

  return true;
}

bool OperationValidationContext::ValidateGru(const mojom::Gru& gru,
                                             OperationId operation_id) {
  if (!IsProcessedOperand(gru.input_operand_id) ||
      !IsProcessedOperand(gru.weight_operand_id) ||
      !IsProcessedOperand(gru.recurrent_weight_operand_id)) {
    return false;
  }
  NoteInputDependency(gru.input_operand_id, operation_id);
  NoteInputDependency(gru.weight_operand_id, operation_id);
  NoteInputDependency(gru.recurrent_weight_operand_id, operation_id);

  const auto* input = GetMojoOperand(gru.input_operand_id);
  const auto* weight = GetMojoOperand(gru.weight_operand_id);
  const auto* recurrent_weight =
      GetMojoOperand(gru.recurrent_weight_operand_id);
  if (!input || !weight || !recurrent_weight) {
    return false;
  }

  const auto& bias_operand_id = gru.bias_operand_id;
  if (bias_operand_id.has_value()) {
    if (!IsProcessedOperand(bias_operand_id.value())) {
      return false;
    }
    NoteInputDependency(bias_operand_id.value(), operation_id);
  }
  const auto& recurrent_bias_operand_id = gru.recurrent_bias_operand_id;
  if (recurrent_bias_operand_id.has_value()) {
    if (!IsProcessedOperand(recurrent_bias_operand_id.value())) {
      return false;
    }
    NoteInputDependency(recurrent_bias_operand_id.value(), operation_id);
  }
  const auto& initial_hidden_state_operand_id =
      gru.initial_hidden_state_operand_id;
  if (initial_hidden_state_operand_id.has_value()) {
    if (!IsProcessedOperand(initial_hidden_state_operand_id.value())) {
      return false;
    }
    NoteInputDependency(initial_hidden_state_operand_id.value(), operation_id);
  }

  for (OperandId output_operand_id : gru.output_operand_ids) {
    if (output_operand_id == gru.input_operand_id ||
        output_operand_id == gru.weight_operand_id ||
        output_operand_id == gru.recurrent_weight_operand_id) {
      return false;
    }
    if (bias_operand_id == output_operand_id ||
        recurrent_bias_operand_id == output_operand_id ||
        initial_hidden_state_operand_id == output_operand_id) {
      return false;
    }
  }

  const base::expected<std::vector<OperandDescriptor>, std::string>
      validated_outputs = ValidateGruAndInferOutput(
          *context_properties_, input->descriptor, weight->descriptor,
          recurrent_weight->descriptor, gru.steps, gru.hidden_size,
          ConvertToGruAttributes(operands_, gru));
  if (!validated_outputs.has_value()) {
    return false;
  }
  if (gru.output_operand_ids.size() != validated_outputs->size()) {
    return false;
  }
  for (size_t i = 0; i < validated_outputs->size(); ++i) {
    const auto* output = GetMojoOperand(gru.output_operand_ids[i]);
    if (!output) {
      return false;
    }
    if (validated_outputs->at(i) != output->descriptor) {
      return false;
    }
  }

  return true;
}

bool OperationValidationContext::ValidateGruCell(const mojom::GruCell& gru_cell,
                                                 OperationId operation_id) {
  if (!IsProcessedOperand(gru_cell.input_operand_id) ||
      !IsProcessedOperand(gru_cell.weight_operand_id) ||
      !IsProcessedOperand(gru_cell.recurrent_weight_operand_id) ||
      !IsProcessedOperand(gru_cell.hidden_state_operand_id)) {
    return false;
  }
  NoteInputDependency(gru_cell.input_operand_id, operation_id);
  NoteInputDependency(gru_cell.weight_operand_id, operation_id);
  NoteInputDependency(gru_cell.recurrent_weight_operand_id, operation_id);
  NoteInputDependency(gru_cell.hidden_state_operand_id, operation_id);

  const mojom::Operand* input = GetMojoOperand(gru_cell.input_operand_id);
  const mojom::Operand* weight = GetMojoOperand(gru_cell.weight_operand_id);
  const mojom::Operand* recurrent_weight =
      GetMojoOperand(gru_cell.recurrent_weight_operand_id);
  const mojom::Operand* hidden_state =
      GetMojoOperand(gru_cell.hidden_state_operand_id);
  if (!input || !weight || !recurrent_weight || !hidden_state) {
    return false;
  }

  const std::optional<OperandId>& bias_operand_id = gru_cell.bias_operand_id;
  if (bias_operand_id.has_value()) {
    if (!IsProcessedOperand(bias_operand_id.value())) {
      return false;
    }
    NoteInputDependency(bias_operand_id.value(), operation_id);
  }
  const std::optional<OperandId>& recurrent_bias_operand_id =
      gru_cell.recurrent_bias_operand_id;
  if (recurrent_bias_operand_id.has_value()) {
    if (!IsProcessedOperand(recurrent_bias_operand_id.value())) {
      return false;
    }
    NoteInputDependency(recurrent_bias_operand_id.value(), operation_id);
  }

  if (gru_cell.output_operand_id == gru_cell.input_operand_id ||
      gru_cell.output_operand_id == gru_cell.weight_operand_id ||
      gru_cell.output_operand_id == gru_cell.recurrent_weight_operand_id ||
      gru_cell.output_operand_id == gru_cell.hidden_state_operand_id ||
      gru_cell.output_operand_id == bias_operand_id ||
      gru_cell.output_operand_id == recurrent_bias_operand_id) {
    return false;
  }

  const base::expected<OperandDescriptor, std::string> validated_output =
      ValidateGruCellAndInferOutput(
          *context_properties_, input->descriptor, weight->descriptor,
          recurrent_weight->descriptor, hidden_state->descriptor,
          gru_cell.hidden_size,
          ConvertToGruCellAttributes(operands_, gru_cell));
  if (!validated_output.has_value()) {
    return false;
  }

  const mojom::Operand* output = GetMojoOperand(gru_cell.output_operand_id);
  if (!output) {
    return false;
  }
  if (validated_output != output->descriptor) {
    return false;
  }

  return true;
}

bool OperationValidationContext::ValidateHardSigmoid(
    const mojom::HardSigmoid& hard_sigmoid,
    OperationId operation_id) {
  if (!ValidateUnaryOperation(
          hard_sigmoid,
          context_properties_->data_type_limits.hard_sigmoid_input,
          operation_id)) {
    return false;
  }
  if (!ValidateHardSigmoidAttributes(hard_sigmoid)) {
    return false;
  }

  return true;
}

bool OperationValidationContext::ValidateLayerNormalization(
    const mojom::LayerNormalization& layer_normalization,
    OperationId operation_id) {
  if (!IsProcessedOperand(layer_normalization.input_operand_id)) {
    return false;
  }
  NoteInputDependency(layer_normalization.input_operand_id, operation_id);

  const auto* input = GetMojoOperand(layer_normalization.input_operand_id);
  const auto* output = GetMojoOperand(layer_normalization.output_operand_id);
  if (!input || !output || output == input) {
    // The layerNormalization operator is invalid.
    return false;
  }

  const auto& scale_operand_id = layer_normalization.scale_operand_id;
  if (scale_operand_id) {
    if (!IsProcessedOperand(*scale_operand_id) ||
        scale_operand_id.value() == layer_normalization.output_operand_id) {
      // The scale operand is invalid.
      return false;
    }
    NoteInputDependency(scale_operand_id.value(), operation_id);
  }
  const auto& bias_operand_id = layer_normalization.bias_operand_id;
  if (bias_operand_id) {
    if (!IsProcessedOperand(bias_operand_id.value()) ||
        bias_operand_id.value() == layer_normalization.output_operand_id) {
      // The bias operand is invalid.
      return false;
    }
    NoteInputDependency(bias_operand_id.value(), operation_id);
  }

  const base::expected<OperandDescriptor, std::string> validated_output =
      ValidateLayerNormalizationAndInferOutput(
          *context_properties_, input->descriptor, layer_normalization.axes,
          ConvertToLayerNormalizationAttributes(operands_,
                                                layer_normalization));
  if (!validated_output.has_value()) {
    return false;
  }
  if (validated_output != output->descriptor) {
    return false;
  }

  return true;
}

bool OperationValidationContext::ValidateLeakyRelu(
    const mojom::LeakyRelu& leaky_relu,
    OperationId operation_id) {
  if (!ValidateUnaryOperation(
          leaky_relu, context_properties_->data_type_limits.leaky_relu_input,
          operation_id)) {
    return false;
  }
  if (!ValidateLeakyReluAttributes(leaky_relu)) {
    return false;
  }

  return true;
}

bool OperationValidationContext::ValidateLinear(const mojom::Linear& linear,
                                                OperationId operation_id) {
  if (!ValidateUnaryOperation(
          linear, context_properties_->data_type_limits.linear_input,
          operation_id)) {
    return false;
  }
  if (!ValidateLinearAttributes(linear)) {
    return false;
  }

  return true;
}

bool OperationValidationContext::ValidateLstm(const mojom::Lstm& lstm,
                                              OperationId operation_id) {
  if (!IsProcessedOperand(lstm.input_operand_id) ||
      !IsProcessedOperand(lstm.weight_operand_id) ||
      !IsProcessedOperand(lstm.recurrent_weight_operand_id)) {
    return false;
  }
  NoteInputDependency(lstm.input_operand_id, operation_id);
  NoteInputDependency(lstm.weight_operand_id, operation_id);
  NoteInputDependency(lstm.recurrent_weight_operand_id, operation_id);

  const auto* input = GetMojoOperand(lstm.input_operand_id);
  const auto* weight = GetMojoOperand(lstm.weight_operand_id);
  const auto* recurrent_weight =
      GetMojoOperand(lstm.recurrent_weight_operand_id);
  if (!input || !weight || !recurrent_weight) {
    return false;
  }

  const auto& bias_operand_id = lstm.bias_operand_id;
  if (bias_operand_id.has_value()) {
    if (!IsProcessedOperand(bias_operand_id.value())) {
      return false;
    }
    NoteInputDependency(bias_operand_id.value(), operation_id);
  }
  const auto& recurrent_bias_operand_id = lstm.recurrent_bias_operand_id;
  if (recurrent_bias_operand_id.has_value()) {
    if (!IsProcessedOperand(recurrent_bias_operand_id.value())) {
      return false;
    }
    NoteInputDependency(recurrent_bias_operand_id.value(), operation_id);
  }
  const auto& peephole_weight_operand_id = lstm.peephole_weight_operand_id;
  if (peephole_weight_operand_id.has_value()) {
    if (!IsProcessedOperand(peephole_weight_operand_id.value())) {
      return false;
    }
    NoteInputDependency(peephole_weight_operand_id.value(), operation_id);
  }
  const auto& initial_hidden_state_operand_id =
      lstm.initial_hidden_state_operand_id;
  if (initial_hidden_state_operand_id.has_value()) {
    if (!IsProcessedOperand(lstm.initial_hidden_state_operand_id.value())) {
      return false;
    }
    NoteInputDependency(initial_hidden_state_operand_id.value(), operation_id);
  }
  const auto& initial_cell_state_operand_id =
      lstm.initial_cell_state_operand_id;
  if (initial_cell_state_operand_id.has_value()) {
    if (!IsProcessedOperand(initial_cell_state_operand_id.value())) {
      return false;
    }
    NoteInputDependency(initial_cell_state_operand_id.value(), operation_id);
  }

  for (OperandId output_operand_id : lstm.output_operand_ids) {
    if (output_operand_id == lstm.input_operand_id ||
        output_operand_id == lstm.weight_operand_id ||
        output_operand_id == lstm.recurrent_weight_operand_id ||
        output_operand_id == lstm.bias_operand_id ||
        output_operand_id == lstm.recurrent_bias_operand_id ||
        output_operand_id == lstm.peephole_weight_operand_id ||
        output_operand_id == lstm.initial_hidden_state_operand_id ||
        output_operand_id == lstm.initial_cell_state_operand_id) {
      return false;
    }
  }

  const base::expected<std::vector<OperandDescriptor>, std::string>
      validated_outputs = ValidateLstmAndInferOutput(
          *context_properties_, input->descriptor, weight->descriptor,
          recurrent_weight->descriptor, lstm.steps, lstm.hidden_size,
          ConvertToLstmAttributes(operands_, lstm));
  if (!validated_outputs.has_value()) {
    return false;
  }
  if (lstm.output_operand_ids.size() != validated_outputs->size()) {
    return false;
  }
  for (size_t i = 0; i < validated_outputs->size(); ++i) {
    const auto* output = GetMojoOperand(lstm.output_operand_ids[i]);
    if (!output) {
      return false;
    }
    if (validated_outputs->at(i) != output->descriptor) {
      return false;
    }
  }

  return true;
}

bool OperationValidationContext::ValidateLstmCell(
    const mojom::LstmCell& lstm_cell,
    OperationId operation_id) {
  if (!IsProcessedOperand(lstm_cell.input_operand_id) ||
      !IsProcessedOperand(lstm_cell.weight_operand_id) ||
      !IsProcessedOperand(lstm_cell.recurrent_weight_operand_id) ||
      !IsProcessedOperand(lstm_cell.hidden_state_operand_id) ||
      !IsProcessedOperand(lstm_cell.cell_state_operand_id)) {
    return false;
  }
  NoteInputDependency(lstm_cell.input_operand_id, operation_id);
  NoteInputDependency(lstm_cell.weight_operand_id, operation_id);
  NoteInputDependency(lstm_cell.recurrent_weight_operand_id, operation_id);
  NoteInputDependency(lstm_cell.hidden_state_operand_id, operation_id);
  NoteInputDependency(lstm_cell.cell_state_operand_id, operation_id);

  const mojom::Operand* input = GetMojoOperand(lstm_cell.input_operand_id);
  const mojom::Operand* weight = GetMojoOperand(lstm_cell.weight_operand_id);
  const mojom::Operand* recurrent_weight =
      GetMojoOperand(lstm_cell.recurrent_weight_operand_id);
  const mojom::Operand* hidden_state =
      GetMojoOperand(lstm_cell.hidden_state_operand_id);
  const mojom::Operand* cell_state =
      GetMojoOperand(lstm_cell.cell_state_operand_id);
  if (!input || !weight || !recurrent_weight || !hidden_state || !cell_state) {
    return false;
  }

  const std::optional<OperandId> bias_operand_id = lstm_cell.bias_operand_id;
  if (bias_operand_id.has_value()) {
    if (!IsProcessedOperand(bias_operand_id.value())) {
      return false;
    }
    NoteInputDependency(bias_operand_id.value(), operation_id);
  }
  const std::optional<OperandId> recurrent_bias_operand_id =
      lstm_cell.recurrent_bias_operand_id;
  if (recurrent_bias_operand_id.has_value()) {
    if (!IsProcessedOperand(recurrent_bias_operand_id.value())) {
      return false;
    }
    NoteInputDependency(recurrent_bias_operand_id.value(), operation_id);
  }
  const std::optional<OperandId> peephole_weight_operand_id =
      lstm_cell.peephole_weight_operand_id;
  if (peephole_weight_operand_id.has_value()) {
    if (!IsProcessedOperand(peephole_weight_operand_id.value())) {
      return false;
    }
    NoteInputDependency(peephole_weight_operand_id.value(), operation_id);
  }

  for (OperandId output_operand_id : lstm_cell.output_operand_ids) {
    if (output_operand_id == lstm_cell.input_operand_id ||
        output_operand_id == lstm_cell.weight_operand_id ||
        output_operand_id == lstm_cell.recurrent_weight_operand_id ||
        output_operand_id == lstm_cell.hidden_state_operand_id ||
        output_operand_id == lstm_cell.cell_state_operand_id ||
        output_operand_id == lstm_cell.bias_operand_id ||
        output_operand_id == lstm_cell.recurrent_bias_operand_id ||
        output_operand_id == lstm_cell.peephole_weight_operand_id) {
      return false;
    }
  }

  const base::expected<std::vector<webnn::OperandDescriptor>, std::string>
      validated_outputs = ValidateLstmCellAndInferOutput(
          *context_properties_, input->descriptor, weight->descriptor,
          recurrent_weight->descriptor, hidden_state->descriptor,
          cell_state->descriptor, lstm_cell.hidden_size,
          ConvertToLstmCellAttributes(operands_, lstm_cell));
  if (!validated_outputs.has_value()) {
    return false;
  }
  if (lstm_cell.output_operand_ids.size() != validated_outputs->size()) {
    return false;
  }
  for (size_t i = 0; i < validated_outputs->size(); ++i) {
    const mojom::Operand* output =
        GetMojoOperand(lstm_cell.output_operand_ids[i]);
    if (!output) {
      return false;
    }
    if (validated_outputs->at(i) != output->descriptor) {
      return false;
    }
  }

  return true;
}

bool OperationValidationContext::ValidateInstanceNormalization(
    const mojom::InstanceNormalization& instance_normalization,
    OperationId operation_id) {
  if (!IsProcessedOperand(instance_normalization.input_operand_id)) {
    return false;
  }
  NoteInputDependency(instance_normalization.input_operand_id, operation_id);

  const auto* input = GetMojoOperand(instance_normalization.input_operand_id);
  const auto* output = GetMojoOperand(instance_normalization.output_operand_id);
  if (!input || !output || output == input) {
    // The instanceNormalization operator is invalid.
    return false;
  }
  const auto& scale_operand_id = instance_normalization.scale_operand_id;
  if (scale_operand_id) {
    if (!IsProcessedOperand(scale_operand_id.value()) ||
        scale_operand_id.value() == instance_normalization.output_operand_id) {
      // The scale operand is invalid.
      return false;
    }
    NoteInputDependency(scale_operand_id.value(), operation_id);
  }
  const auto& bias_operand_id = instance_normalization.bias_operand_id;
  if (bias_operand_id) {
    if (!IsProcessedOperand(bias_operand_id.value()) ||
        bias_operand_id.value() == instance_normalization.output_operand_id) {
      // The bias operand is invalid.
      return false;
    }
    NoteInputDependency(bias_operand_id.value(), operation_id);
  }

  const base::expected<OperandDescriptor, std::string> validated_output =
      ValidateInstanceNormalizationAndInferOutput(
          *context_properties_, input->descriptor,
          ConvertToInstanceNormalizationAttributes(
              *context_properties_, operands_, instance_normalization));
  if (!validated_output.has_value()) {
    return false;
  }
  if (validated_output != output->descriptor) {
    return false;
  }

  return true;
}

bool OperationValidationContext::ValidateMatmul(const mojom::Matmul& matmul,
                                                OperationId operation_id) {
  if (!IsProcessedOperand(matmul.a_operand_id) ||
      !IsProcessedOperand(matmul.b_operand_id)) {
    return false;
  }
  NoteInputDependency(matmul.a_operand_id, operation_id);
  NoteInputDependency(matmul.b_operand_id, operation_id);

  auto* a = GetMojoOperand(matmul.a_operand_id);
  auto* b = GetMojoOperand(matmul.b_operand_id);
  auto* output = GetMojoOperand(matmul.output_operand_id);
  if (!a || !b || !output || output == a || output == b) {
    // The matmul operator is invalid.
    return false;
  }
  const base::expected<OperandDescriptor, std::string> validated_output =
      ValidateMatmulAndInferOutput(*context_properties_, a->descriptor,
                                   b->descriptor, matmul.label);
  if (!validated_output.has_value()) {
    return false;
  }
  if (validated_output != output->descriptor) {
    return false;
  }

  return true;
}

bool OperationValidationContext::ValidatePad(const mojom::Pad& pad,
                                             OperationId operation_id) {
  if (!IsProcessedOperand(pad.input_operand_id)) {
    return false;
  }
  NoteInputDependency(pad.input_operand_id, operation_id);

  auto* input = GetMojoOperand(pad.input_operand_id);
  auto* output = GetMojoOperand(pad.output_operand_id);
  if (!input || !output || output == input) {
    // The pad operator is invalid.
    return false;
  }

  const base::expected<OperandDescriptor, std::string> validated_output =
      ValidatePadAndInferOutput(*context_properties_, input->descriptor,
                                pad.beginning_padding, pad.ending_padding,
                                FromMojoPaddingMode(pad.mode->which()),
                                pad.label);
  if (!validated_output.has_value()) {
    return false;
  }
  if (validated_output != output->descriptor) {
    return false;
  }

  return true;
}

bool OperationValidationContext::ValidatePool2d(const mojom::Pool2d& pool2d,
                                                OperationId operation_id) {
  if (!IsProcessedOperand(pool2d.input_operand_id)) {
    return false;
  }
  NoteInputDependency(pool2d.input_operand_id, operation_id);

  auto* input = GetMojoOperand(pool2d.input_operand_id);
  auto* output = GetMojoOperand(pool2d.output_operand_id);
  if (!input || !output || output == input) {
    // The pool2d operator is invalid.
    return false;
  }

  if (output->descriptor.Rank() != 4) {
    return false;
  }
  const base::expected<OperandDescriptor, std::string> validated_output =
      ValidatePool2dAndInferOutput(
          *context_properties_, input->descriptor,
          ConvertToPool2dAttributes(*context_properties_, pool2d, output),
          FromMojoPool2dType(pool2d.kind));
  if (!validated_output.has_value()) {
    return false;
  }
  if (validated_output != output->descriptor) {
    return false;
  }

  return true;
}

bool OperationValidationContext::ValidatePrelu(const mojom::Prelu& prelu,
                                               OperationId operation_id) {
  if (!IsProcessedOperand(prelu.input_operand_id) ||
      !IsProcessedOperand(prelu.slope_operand_id)) {
    return false;
  }
  NoteInputDependency(prelu.input_operand_id, operation_id);
  NoteInputDependency(prelu.slope_operand_id, operation_id);

  auto* input = GetMojoOperand(prelu.input_operand_id);
  auto* output = GetMojoOperand(prelu.output_operand_id);
  auto* slope = GetMojoOperand(prelu.slope_operand_id);
  if (!input || !output || !slope || output == input || output == slope) {
    // The prelu operator is invalid.
    return false;
  }

  const base::expected<OperandDescriptor, std::string> validated_output =
      ValidatePreluAndInferOutput(*context_properties_, input->descriptor,
                                  slope->descriptor, prelu.label);
  if (!validated_output.has_value()) {
    return false;
  }
  if (validated_output != output->descriptor) {
    return false;
  }

  return true;
}

bool OperationValidationContext::ValidateQuantizeLinear(
    const mojom::QuantizeLinear& quantize_linear,
    OperationId operation_id) {
  if (!IsProcessedOperand(quantize_linear.input_operand_id) ||
      !IsProcessedOperand(quantize_linear.scale_operand_id) ||
      !IsProcessedOperand(quantize_linear.zero_point_operand_id)) {
    return false;
  }
  NoteInputDependency(quantize_linear.input_operand_id, operation_id);
  NoteInputDependency(quantize_linear.scale_operand_id, operation_id);
  NoteInputDependency(quantize_linear.zero_point_operand_id, operation_id);

  auto* input = GetMojoOperand(quantize_linear.input_operand_id);
  auto* output = GetMojoOperand(quantize_linear.output_operand_id);
  auto* scale = GetMojoOperand(quantize_linear.scale_operand_id);
  auto* zero_point = GetMojoOperand(quantize_linear.zero_point_operand_id);
  if (!input || !output || !scale || !zero_point || output == input ||
      output == scale || output == zero_point) {
    // The quantize_linear operator is invalid.
    return false;
  }

  const base::expected<OperandDescriptor, std::string> validated_output =
      ValidateQuantizeLinearAndInferOutput(
          *context_properties_, input->descriptor, scale->descriptor,
          zero_point->descriptor, quantize_linear.label);
  if (!validated_output.has_value()) {
    return false;
  }
  if (validated_output != output->descriptor) {
    return false;
  }

  return true;
}

bool OperationValidationContext::ValidateResample2d(
    const mojom::Resample2d& resample2d,
    OperationId operation_id) {
  if (!IsProcessedOperand(resample2d.input_operand_id)) {
    return false;
  }
  NoteInputDependency(resample2d.input_operand_id, operation_id);

  auto* input = GetMojoOperand(resample2d.input_operand_id);
  auto* output = GetMojoOperand(resample2d.output_operand_id);
  if (!input || !output || output == input) {
    // The resample2d operator is invalid.
    return false;
  }

  // Validate and infer the output for resample2d with given scales or with
  // the sizes from output dimensions along axes.
  std::variant<base::span<const float>, base::span<const uint32_t>>
      scales_or_sizes;
  const auto& axes = resample2d.axes;
  std::vector<uint32_t> sizes;
  const auto& output_dimensions = output->descriptor.shape();
  if (axes.size() != 2 || axes[0] >= output_dimensions.size() ||
      axes[1] >= output_dimensions.size()) {
    return false;
  }
  const std::array<uint32_t, 2> kResample2dChannelFirstAxes{2u, 3u};
  const std::array<uint32_t, 2> kResample2dChannelLastAxes{1u, 2u};
  switch (context_properties_->resample_2d_axes) {
    case Resample2DAxes::kAny:
      break;
    case Resample2DAxes::kChannelsFirst:
      if (!std::ranges::equal(axes, kResample2dChannelFirstAxes)) {
        return false;
      }
      break;
    case Resample2DAxes::kChannelsLast:
      if (!std::ranges::equal(axes, kResample2dChannelLastAxes)) {
        return false;
      }
      break;
  }
  if (resample2d.scales) {
    scales_or_sizes = resample2d.scales.value();
  } else {
    sizes = {output_dimensions[axes[0]], output_dimensions[axes[1]]};
    scales_or_sizes = sizes;
  }

  const base::expected<OperandDescriptor, std::string> validated_output =
      ValidateResample2dAndInferOutput(*context_properties_, input->descriptor,
                                       scales_or_sizes, axes, resample2d.label);
  if (!validated_output.has_value()) {
    return false;
  }
  if (validated_output != output->descriptor) {
    return false;
  }

  return true;
}

bool OperationValidationContext::ValidateReshape(const mojom::Reshape& reshape,
                                                 OperationId operation_id) {
  if (!IsProcessedOperand(reshape.input_operand_id)) {
    return false;
  }
  NoteInputDependency(reshape.input_operand_id, operation_id);

  auto* input = GetMojoOperand(reshape.input_operand_id);
  auto* output = GetMojoOperand(reshape.output_operand_id);
  if (!input || !output || output == input) {
    // The reshape operator is invalid.
    return false;
  }
  if (!context_properties_->data_type_limits.reshape_input.Supports(
          input->descriptor)) {
    return false;
  }
  if (!context_properties_->data_type_limits.reshape_input.ranks.Supports(
          output->descriptor.Rank())) {
    return false;
  }
  if (output->descriptor.data_type() != input->descriptor.data_type()) {
    return false;
  }

  if (input->descriptor.NumberOfElements() !=
      output->descriptor.NumberOfElements()) {
    // The output shape is not expected.
    return false;
  }
  return true;
}

bool OperationValidationContext::ValidateReverseOperation(
    const mojom::Reverse& reverse,
    OperationId operation_id) {
  if (!IsProcessedOperand(reverse.input_operand_id)) {
    return false;
  }
  NoteInputDependency(reverse.input_operand_id, operation_id);

  auto* input = GetMojoOperand(reverse.input_operand_id);
  auto* output = GetMojoOperand(reverse.output_operand_id);
  if (!input || !output || output == input) {
    return false;
  }

  const base::expected<OperandDescriptor, std::string> validated_output =
      ValidateReverseAndInferOutput(*context_properties_, input->descriptor,
                                    reverse.axes, reverse.label);
  if (!validated_output.has_value()) {
    return false;
  }
  if (validated_output != output->descriptor) {
    return false;
  }

  return true;
}

bool OperationValidationContext::ValidateScatterElements(
    const mojom::ScatterElements& scatter_elements,
    OperationId operation_id) {
  if (!IsProcessedOperand(scatter_elements.input_operand_id) ||
      !IsProcessedOperand(scatter_elements.indices_operand_id) ||
      !IsProcessedOperand(scatter_elements.updates_operand_id)) {
    return false;
  }
  NoteInputDependency(scatter_elements.input_operand_id, operation_id);
  NoteInputDependency(scatter_elements.indices_operand_id, operation_id);
  NoteInputDependency(scatter_elements.updates_operand_id, operation_id);

  auto* input = GetMojoOperand(scatter_elements.input_operand_id);
  auto* indices = GetMojoOperand(scatter_elements.indices_operand_id);
  auto* updates = GetMojoOperand(scatter_elements.updates_operand_id);
  auto* output = GetMojoOperand(scatter_elements.output_operand_id);
  if (!input || !indices || !updates || !output || output == input ||
      output == indices || output == updates) {
    return false;
  }

  const base::expected<OperandDescriptor, std::string> validated_output =
      ValidateScatterElementsAndInferOutput(
          *context_properties_, input->descriptor, indices->descriptor,
          updates->descriptor, scatter_elements.axis, scatter_elements.label);
  if (!validated_output.has_value()) {
    return false;
  }
  if (validated_output != output->descriptor) {
