/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "op_map.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tim/vx/ops/activations.h"
#include "tim/vx/ops/addn.h"
#include "tim/vx/ops/batch2space.h"
#include "tim/vx/ops/concat.h"
#include "tim/vx/ops/conv2d.h"
#include "tim/vx/ops/depth2space.h"
#include "tim/vx/ops/elementwise.h"
#include "tim/vx/ops/fullyconnected.h"
#include "tim/vx/ops/gather.h"
#include "tim/vx/ops/gathernd.h"
#include "tim/vx/ops/l2normalization.h"
#include "tim/vx/ops/localresponsenormalization.h"
#include "tim/vx/ops/logical.h"
#include "tim/vx/ops/pad.h"
#include "tim/vx/ops/pool2d.h"
#include "tim/vx/ops/reduce.h"
#include "tim/vx/ops/reshape.h"
#include "tim/vx/ops/resize.h"
#include "tim/vx/ops/reverse.h"
#include "tim/vx/ops/simple_operations.h"
#include "tim/vx/ops/slice.h"
#include "tim/vx/ops/softmax.h"
#include "tim/vx/ops/space2batch.h"
#include "tim/vx/ops/space2depth.h"
#include "tim/vx/ops/split.h"
#include "tim/vx/ops/stridedslice.h"
#include "tim/vx/ops/transpose.h"
#include "utils.h"

namespace {

inline tim::vx::PadType TflitePadTypeToVsiPadType(TfLitePadding pad) {
  switch (pad) {
    case kTfLitePaddingUnknown:
      return tim::vx::PadType::AUTO;
    case kTfLitePaddingValid:
      return tim::vx::PadType::VALID;
    case kTfLitePaddingSame:
      return tim::vx::PadType::SAME;
    default:
      LOG(ERROR) << "Unsuppoted pad type:" << pad;
      break;
  }

  return tim::vx::PadType::AUTO;
}

/// Insert activation layer before the `original_tensor`
/// Return the input tensor of new activation layer
std::shared_ptr<tim::vx::Tensor> ProcessFusedActivation(
    vx::delegate::Delegate* delegate,
    TfLiteFusedActivation fused_activation,
    const std::shared_ptr<tim::vx::Tensor>& original_tensor) {
  std::shared_ptr<tim::vx::Operation> op = nullptr;
  switch (fused_activation) {
    case kTfLiteActNone:
      return original_tensor;
    case kTfLiteActRelu:
      op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Relu>();
      break;
    case kTfLiteActRelu1:
      op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Relu1>();
      break;
    case kTfLiteActRelu6:
      op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Relu6>();
      break;
    case kTfLiteActTanh:
      op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Tanh>();
      break;
    case kTfLiteActSigmoid:
      op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Sigmoid>();
      break;
    default:
      LOG(WARNING) << "Unsupported fused activation: " << fused_activation;
  }

  auto processed_tensor = delegate->GetGraph()->CreateTensor(
      original_tensor->GetSpec().AsTransientSpec());

  (*op).BindInput(processed_tensor);
  (*op).BindOutput(original_tensor);

  delegate->GetOps().push_back(op);
  delegate->GetTensors().push_back(processed_tensor);

  return processed_tensor;
}

// Insert a transpose node after the `original_tensor`
std::shared_ptr<tim::vx::Tensor> TransposeInputTensor(
    vx::delegate::Delegate* delegate,
    const std::shared_ptr<tim::vx::Tensor>& original_tensor,
    const std::vector<uint32_t>& perm) {
  auto transposed_tensor_spec = original_tensor->GetSpec().AsTransientSpec();
  if (transposed_tensor_spec.quantization_.Type() ==
      tim::vx::QuantType::SYMMETRIC_PER_CHANNEL) {
    int32_t new_channel_dim = vx::delegate::utils::MapChannelDim(
        perm, transposed_tensor_spec.quantization_.ChannelDim());
    transposed_tensor_spec.quantization_.SetChannelDim(new_channel_dim);
  }

  auto transposed_tensor =
      delegate->GetGraph()->CreateTensor(transposed_tensor_spec);

  std::shared_ptr<tim::vx::Operation> op =
      delegate->GetGraph()->CreateOperation<tim::vx::ops::Transpose>(perm);
  (*op).BindInput(original_tensor);
  (*op).BindOutput(transposed_tensor);

  delegate->GetOps().push_back(op);
  delegate->GetTensors().push_back(transposed_tensor);

  return transposed_tensor;
}

// Insert a transpose node before the `original_tensor`
std::shared_ptr<tim::vx::Tensor> TransposeOutputTensor(
    vx::delegate::Delegate* delegate,
    const std::shared_ptr<tim::vx::Tensor>& original_tensor,
    const std::vector<uint32_t>& perm) {
  auto transposed_tensor = delegate->GetGraph()->CreateTensor(
      original_tensor->GetSpec().AsTransientSpec());

  std::shared_ptr<tim::vx::Operation> op =
      delegate->GetGraph()->CreateOperation<tim::vx::ops::Transpose>(perm);
  (*op).BindInput(transposed_tensor);
  (*op).BindOutput(original_tensor);

  delegate->GetOps().push_back(op);
  delegate->GetTensors().push_back(transposed_tensor);

  return transposed_tensor;
}

std::shared_ptr<tim::vx::Tensor> ReverseInputTensor(
    vx::delegate::Delegate* delegate,
    const std::shared_ptr<tim::vx::Tensor>& original_tensor,
    int32_t* axis,
    uint32_t axis_num) {
  auto reversed_tensor = delegate->GetGraph()->CreateTensor(
      original_tensor->GetSpec().AsTransientSpec());
  std::shared_ptr<tim::vx::Operation> op =
      delegate->GetGraph()->CreateOperation<tim::vx::ops::Reverse>(axis,
                                                                   axis_num);
  (*op).BindInput(original_tensor);
  (*op).BindOutput(reversed_tensor);

  delegate->GetOps().push_back(op);
  delegate->GetTensors().push_back(reversed_tensor);

  return reversed_tensor;
}

enum class ActionTargetType { INPUT, OUTPUT, STATE };

struct IAction {
  virtual ActionTargetType GetActionTargetType() const = 0;
  virtual bool process(vx::delegate::Delegate* delegate,
                       std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                       std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                       std::vector<std::shared_ptr<tim::vx::Tensor>>& states,
                       const void* params) const = 0;
};

template <ActionTargetType type, int Port>
struct ActionBase : public IAction {
  ActionTargetType type_{type};
  int port_{Port};
  bool process(vx::delegate::Delegate* delegate,
               std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
               std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
               std::vector<std::shared_ptr<tim::vx::Tensor>>& states,
               const void* params) const override {
    return true;
  }
  ActionTargetType GetActionTargetType() const final { return type_; }
};

template <int Port, uint32_t... TransposeVec>
struct TransposeInputAction : public ActionBase<ActionTargetType::INPUT, Port> {
  std::vector<uint32_t> perm_{TransposeVec...};
  bool process(vx::delegate::Delegate* delegate,
               std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
               std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
               std::vector<std::shared_ptr<tim::vx::Tensor>>& states,
               const void* params) const final {
    inputs[this->port_] =
        TransposeInputTensor(delegate, inputs[this->port_], perm_);
    return true;
  }
};

template <int Port, uint32_t... TransposeVec>
struct TransposeOutputAction
    : public ActionBase<ActionTargetType::OUTPUT, Port> {
  std::vector<uint32_t> perm_{TransposeVec...};
  bool process(vx::delegate::Delegate* delegate,
               std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
               std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
               std::vector<std::shared_ptr<tim::vx::Tensor>>& states,
               const void* params) const final {
    outputs[this->port_] =
        TransposeOutputTensor(delegate, outputs[this->port_], perm_);
    return true;
  }
};

template <int Port, typename T_Param>
struct FusedActivationAction
    : public ActionBase<ActionTargetType::OUTPUT, Port> {
  bool process(vx::delegate::Delegate* delegate,
               std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
               std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
               std::vector<std::shared_ptr<tim::vx::Tensor>>& states,
               const void* params) const final {
    const auto builtin = reinterpret_cast<const T_Param*>(params);
    outputs[this->port_] = ProcessFusedActivation(
        delegate, builtin->activation, outputs[this->port_]);
    return true;
  }
};

template <typename T_Param, typename... Actions>
struct OpMapperBase : public vx::op_map::IOpMapper {
  std::vector<std::unique_ptr<IAction>> actions_;

  OpMapperBase() {
    (void)std::initializer_list<int>{
        0, (actions_.emplace_back(std::make_unique<Actions>()), 0)...};
  }

  size_t GetParamSize() const override { return sizeof(T_Param); }

  bool IsSupported(TfLiteContext* context,
                   TfLiteNode* node,
                   const TfLiteRegistration* registration) const override {
    for (int i = 0; i < node->inputs->size; i++) {
      int input_index = node->inputs->data[i];
      if (context->tensors[input_index].type == kTfLiteInt16) {
        LOG(ERROR) << "Int16 input is not supported";
        return false;
      }
      if (context->tensors[input_index].type == kTfLiteInt64) {
        LOG(ERROR) << "Int64 input is not supported";
        return false;
      }
      for (int j = 0; j < context->tensors[input_index].dims->size; j++) {
        if (context->tensors[input_index].dims->data[j] == 0) {
          LOG(ERROR) << "vx delegate doesn't support the tensor has 0 dim";
          return false;
        }
      }
    }
    for (int i = 0; i < node->outputs->size; i++) {
      int output_index = node->outputs->data[i];
      if (context->tensors[output_index].type == kTfLiteInt16) {
        LOG(ERROR) << "Int16 output is not supported";
        return false;
      }
      if (context->tensors[output_index].type == kTfLiteInt64) {
        LOG(ERROR) << "Int64 output is not supported";
        return false;
      }
      for (int j = 0; j < context->tensors[output_index].dims->size; j++) {
        if (context->tensors[output_index].dims->data[j] == 0) {
          LOG(ERROR) << "vx-delegate doesn't support the tensor has 0 dim";
          return false;
        }
      }
    }

    return IsOpSupported(context, node, registration);
  }

  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    return true;
  }

  bool MapOp(vx::delegate::Delegate* delegate,
             std::vector<std::shared_ptr<tim::vx::Tensor>> inputs,
             std::vector<std::shared_ptr<tim::vx::Tensor>> outputs,
             std::vector<std::shared_ptr<tim::vx::Tensor>> states,
             const void* params) {
    bool status = true;

    for (auto& a : actions_) {
      if (a->GetActionTargetType() == ActionTargetType::INPUT) {
        a->process(delegate, inputs, outputs, states, params);
      }
    }

    for (auto it = actions_.rbegin(); it != actions_.rend(); it++) {
      if ((*it)->GetActionTargetType() == ActionTargetType::OUTPUT) {
        (*it)->process(delegate, inputs, outputs, states, params);
      }
    }

    status = HandleMapOp(delegate, inputs, outputs, states, params);

    return status;
  }

  virtual bool HandleMapOp(
      vx::delegate::Delegate* delegate,
      std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
      std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
      std::vector<std::shared_ptr<tim::vx::Tensor>>& states,
      const void* params) {
    return HandleMapOp(delegate, inputs, outputs, params);
  }

  virtual bool HandleMapOp(
      vx::delegate::Delegate* delegate,
      std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
      std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
      const void* params) {
    return false;
  }
};

}  // namespace
namespace vx {
namespace op_map {

template <typename T_OperationType>
struct SimpleOpMapper : public OpMapperBase<EmptyStructPlaceholder> {
  std::string name_;

  SimpleOpMapper(std::string name) : name_(name) {}

  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    LOG(INFO) << "Creating " << name_ << " op";

    auto op = delegate->GetGraph()->CreateOperation<T_OperationType>();
    (*op).BindInputs(inputs).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

template <typename T_OperationType, typename T_Param>
struct SimpleOpWithFusedActiovationMapper
    : public OpMapperBase<T_Param, FusedActivationAction<0, T_Param>> {
  std::string name_;

  SimpleOpWithFusedActiovationMapper(std::string name) : name_(name) {}

  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    LOG(INFO) << "Creating " << name_ << " op";

    auto op = delegate->GetGraph()->CreateOperation<T_OperationType>();
    (*op).BindInputs(inputs).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

template <typename T_Param>
struct Conv2dKind : public OpMapperBase<T_Param,
                                        TransposeInputAction<0, 1, 2, 0, 3>,
                                        FusedActivationAction<0, T_Param>,
                                        TransposeOutputAction<0, 2, 0, 1, 3>> {
};

struct FullyConnectedMapper
    : public OpMapperBase<
          TfLiteFullyConnectedParams,
          FusedActivationAction<0, TfLiteFullyConnectedParams>> {
  bool IsOpSupported(TfLiteContext* context,
                     TfLiteNode* node,
                     const TfLiteRegistration* registration) const override {
    const auto builtin =
        reinterpret_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);
    if (builtin->weights_format ==
        kTfLiteFullyConnectedWeightsFormatShuffled4x16Int8) {
      LOG(ERROR) << "Shuffled weight is not supported";
      return false;
    }
    for (int i = 0; i < node->inputs->size; i++) {
      int input_index = node->inputs->data[i];
      if (context->tensors[input_index].type == kTfLiteInt16) {
        LOG(ERROR) << "Int16 input is not supported";
        return false;
      }
    }
    return true;
  }

  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    LOG(INFO) << "Creating fully connected op";
    const auto builtin =
        reinterpret_cast<const TfLiteFullyConnectedParams*>(params);
    auto input_tensor = inputs[0];
    auto weight_tensor = inputs[1];

    if (input_tensor->GetShape().size() > 2 ||
        (input_tensor->GetShape().size() == 2 &&
         input_tensor->GetShape()[0] != weight_tensor->GetShape()[0])) {
      uint32_t input_size = weight_tensor->GetShape()[0];
      uint32_t total_input_size = 1;
      for (int i = 0; i < input_tensor->GetShape().size(); i++) {
        total_input_size *= input_tensor->GetShape()[i];
      }
      uint32_t input_batch = total_input_size / input_size;
      auto reshape_output = delegate->GetGraph()->CreateTensor(
          input_tensor->GetSpec().AsTransientSpec());
      std::vector<uint32_t> new_shape{input_size, input_batch};
      auto reshape_op =
          delegate->GetGraph()->CreateOperation<tim::vx::ops::Reshape>(
              new_shape);
      (*reshape_op).BindInput(inputs[0]);
      (*reshape_op).BindOutput(reshape_output);
      delegate->GetOps().push_back(reshape_op);
      inputs[0] = reshape_output;
    }

    auto op =
        delegate->GetGraph()->CreateOperation<tim::vx::ops::FullyConnected>(
            1, weight_tensor->GetShape()[1]);
    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct SoftmaxMapper : public OpMapperBase<TfLiteSoftmaxParams> {
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    LOG(INFO) << "Creating softmax op";
    auto builtin = reinterpret_cast<const TfLiteSoftmaxParams*>(params);
    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Softmax>(
        builtin->beta, 0);
    (*op).BindInputs(inputs).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct Conv2dMapper : public Conv2dKind<TfLiteConvParams> {
  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    return true;
  }

  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    LOG(INFO) << "Creating Conv2d op";
    const auto builtin = reinterpret_cast<const TfLiteConvParams*>(params);

    uint32_t weights = inputs[1]->GetShape()[3];
    uint32_t kernel_h = inputs[1]->GetShape()[1];
    uint32_t kernel_w = inputs[1]->GetShape()[0];
    if (!inputs[1]->IsConstTensor()) {
      weights = inputs[1]->GetShape()[3];
      kernel_h = inputs[1]->GetShape()[2];
      kernel_w = inputs[1]->GetShape()[1];
      inputs[1] = TransposeInputTensor(delegate, inputs[1], {1, 2, 0, 3});
    }

    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Conv2d>(
        static_cast<int32_t>(weights),
        TflitePadTypeToVsiPadType(builtin->padding),
        std::array<uint32_t, 2>({kernel_w, kernel_h}),
        std::array<uint32_t, 2>(
            {builtin->stride_width, builtin->stride_height}),
        std::array<uint32_t, 2>(
            {builtin->dilation_width_factor, builtin->dilation_height_factor}));

    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

template <tim::vx::PoolType poolType>
struct Pool2dMapper : public Conv2dKind<TfLitePoolParams> {
  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    return true;
  }

  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    LOG(INFO) << "Creating Pool2d(" << static_cast<int>(poolType) << ") op";
    const auto builtin = reinterpret_cast<const TfLitePoolParams*>(params);

    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Pool2d>(
        poolType,
        TflitePadTypeToVsiPadType(builtin->padding),
        std::array<uint32_t, 2>(
            {builtin->filter_width, builtin->filter_height}),
        std::array<uint32_t, 2>(
            {builtin->stride_width, builtin->stride_height}));

    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct DepthwiseConv2dMapper : public Conv2dKind<TfLiteDepthwiseConvParams> {
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    LOG(INFO) << "Creating DepthwiseConv2d op";
    const auto builtin =
        reinterpret_cast<const TfLiteDepthwiseConvParams*>(params);

    uint32_t weights = inputs[1]->GetShape()[2];
    uint32_t kernel_h = inputs[1]->GetShape()[1];
    uint32_t kernel_w = inputs[1]->GetShape()[0];
    if (!inputs[1]->IsConstTensor()) {
      weights = inputs[1]->GetShape()[0];
      kernel_h = inputs[1]->GetShape()[2];
      kernel_w = inputs[1]->GetShape()[1];
      inputs[1] = TransposeInputTensor(delegate, inputs[1], {1, 2, 0, 3});
    }

    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Conv2d>(
        static_cast<int32_t>(weights),
        TflitePadTypeToVsiPadType(builtin->padding),
        std::array<uint32_t, 2>({kernel_w, kernel_h}),
        std::array<uint32_t, 2>(
            {builtin->stride_width, builtin->stride_height}),
        std::array<uint32_t, 2>(
            {builtin->dilation_width_factor, builtin->dilation_height_factor}),
        builtin->depth_multiplier);

    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));
  }
};

struct ConcatenationMapper
    : public OpMapperBase<TfLiteConcatenationParams,
                          FusedActivationAction<0, TfLiteConcatenationParams>> {
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    LOG(INFO) << "Creating Concatenation op";
    const auto builtin =
        reinterpret_cast<const TfLiteConcatenationParams*>(params);
    auto output_tensor = outputs[0];

    auto axis = vx::delegate::utils::ConvertAxis(builtin->axis,
                                                 inputs[0]->GetShape().size());

    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Concat>(
        axis, inputs.size());

    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));
  }
};

struct LocalResponseNormalizationMapper
    : public OpMapperBase<TfLiteLocalResponseNormParams> {
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    LOG(INFO) << "Creating LRN op";
    const auto builtin =
        reinterpret_cast<const TfLiteLocalResponseNormParams*>(params);
    auto op = delegate->GetGraph()
                  ->CreateOperation<tim::vx::ops::LocalResponseNormalization>(
                      builtin->radius,
                      builtin->alpha,
                      builtin->beta,
                      builtin->bias,
                      0);
    (*op).BindInputs(inputs).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct L2NormalizationMapper
    : public OpMapperBase<TfLiteL2NormParams,
                          FusedActivationAction<0, TfLiteL2NormParams>> {
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    LOG(INFO) << "Creating L2Normaliztion op";
    const auto builtin = reinterpret_cast<const TfLiteL2NormParams*>(params);

    auto op =
        delegate->GetGraph()->CreateOperation<tim::vx::ops::L2Normalization>(0);

    (*op).BindInputs(inputs).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct ReshapeMapper : public OpMapperBase<TfLiteReshapeParams> {
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    LOG(INFO) << "Creating Reshape op";
    const auto builtin = reinterpret_cast<const TfLiteReshapeParams*>(params);
    std::vector<uint32_t> new_shape;

    // The new shape may be passed to reshape op by
    // builtin prarameters or inputs[1], the two formats should be handled.
    if (inputs.size() == 2 &&
        inputs[1]->GetDataType() == tim::vx::DataType::INT32 &&
        inputs[1]->GetShape().size() == 1) {
      std::vector<int32_t> shape_from_input1(inputs[1]->GetShape()[0]);
      inputs[1]->CopyDataFromTensor(shape_from_input1.data());
      new_shape.assign(shape_from_input1.rbegin(), shape_from_input1.rend());
    } else {
      for (int i = builtin->num_dimensions - 1; i >= 0; i--) {
        new_shape.push_back(static_cast<uint32_t>(builtin->shape[i]));
      }
    }

    auto op =
        delegate->GetGraph()->CreateOperation<tim::vx::ops::Reshape>(new_shape);
    (*op).BindInput(inputs[0]);
    (*op).BindOutput(outputs[0]);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct StridedSliceMapper : public OpMapperBase<TfLiteStridedSliceParams> {
  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    LOG(INFO) << "Check  StridedSlice";
    const auto builtin =
        reinterpret_cast<const TfLiteStridedSliceParams*>(node->builtin_data);
    if (builtin->new_axis_mask) {
      LOG(ERROR) << "new_axis_mask > 0 is not supported";
      return false;
    }
    if (builtin->ellipsis_mask) {
      LOG(ERROR) << "ellipsis_mask > 0 is not supported";
      return false;
    }
    return true;
  }

  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    LOG(INFO) << "Creating StridedSlice op";
    const auto builtin =
        reinterpret_cast<const TfLiteStridedSliceParams*>(params);
    auto input_tensor = inputs[0];
    auto begin_tensor = inputs[1];
    auto end_tensor = inputs[2];
    auto strides_tensor = inputs[3];
    auto output_tensor = outputs[0];
    auto const& input_shape = input_tensor->GetShape();
    int begin_mask = builtin->begin_mask;
    int end_mask = builtin->end_mask;
    int ellipsis_mask = builtin->ellipsis_mask;
    int new_axis_mask = builtin->new_axis_mask;
    int shrink_axis_mask = builtin->shrink_axis_mask;

    std::vector<int32_t> begin_dims(begin_tensor->GetShape()[0]);
    begin_tensor->CopyDataFromTensor(begin_dims.data());
    for (size_t i = 0; i < begin_dims.size(); i++) {
      if (begin_mask & (1 << i)) {
        begin_dims[i] = -1;
      }
    }
    std::reverse(begin_dims.begin(), begin_dims.end());

    std::vector<int32_t> end_dims(end_tensor->GetShape()[0]);
    end_tensor->CopyDataFromTensor(end_dims.data());
    for (size_t i = 0; i < end_dims.size(); i++) {
      if (end_mask & (1 << i)) {
        end_dims[i] = -1;
      }
    }
    std::reverse(end_dims.begin(), end_dims.end());

    std::vector<int32_t> strides_dims(strides_tensor->GetShape()[0]);
    strides_tensor->CopyDataFromTensor(strides_dims.data());
    std::reverse(strides_dims.begin(), strides_dims.end());

    if (ellipsis_mask) {
      LOG(WARNING) << "ellipsis_mask > 0 is not supported";
    } else {
      size_t i = begin_dims.size();
      for (; i < input_shape.size(); i++) {
        begin_dims.insert(begin_dims.begin(), -1);
      }
      i = end_dims.size();
      for (; i < input_shape.size(); i++) {
        end_dims.insert(end_dims.begin(), -1);
      }
      i = strides_dims.size();
      for (; i < input_shape.size(); i++) {
        strides_dims.insert(strides_dims.begin(), -1);
      }
    }

    for (size_t i = 0; i < begin_dims.size(); i++) {
      begin_dims[i] = begin_dims[i] == -1 ? 0 : begin_dims[i];
    }

    for (size_t i = 0; i < end_dims.size(); i++) {
      end_dims[i] = end_dims[i] == -1 ? input_shape[i] : end_dims[i];
      end_dims[i] = end_dims[i] > static_cast<int32_t>(input_shape[i])
                        ? input_shape[i]
                        : end_dims[i];
    }

    for (size_t i = 0; i < strides_dims.size(); i++) {
      strides_dims[i] = strides_dims[i] == -1 ? 1 : strides_dims[i];
    }

    begin_mask = 0;
    end_mask = 0;

    if (shrink_axis_mask) {
      int32_t t = 0;
      int32_t input_dim = input_shape.size();
      for (size_t i = 0; i < input_dim; i++) {
        if (shrink_axis_mask & (1 << i)) {
          t = t | (1 << (input_dim - i - 1));
        }
      }
      shrink_axis_mask = t;
    }

    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::StridedSlice>(
        begin_dims,
        end_dims,
        strides_dims,
        begin_mask,
        end_mask,
        shrink_axis_mask);
    (*op).BindInput(input_tensor);
    (*op).BindOutput(output_tensor);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct PadMapper : public OpMapperBase<EmptyStructPlaceholder> {
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    LOG(INFO) << "Creating Pad op";
    auto padding = inputs[1];
    std::vector<uint32_t> padding_shape = padding->GetShape();
    uint32_t pad = 1;
    for (auto s : padding_shape) {
      pad *= s;
    }
    std::vector<uint32_t> pad_size(pad);
    padding->CopyDataFromTensor(pad_size.data());
    std::vector<uint32_t> front_size;
    std::vector<uint32_t> back_size;
    for (int i = pad_size.size() - 1; i >= 0; i -= 2) {
      back_size.push_back(pad_size[i]);
      front_size.push_back(pad_size[i - 1]);
    }
    int32_t val = 0;
    if (inputs.size() > 2) {
      auto pad_value = inputs[2];
      if (!pad_value->IsPlaceHolder()) {
        pad_value->CopyDataFromTensor(&val);
      }
    }
    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Pad>(
        front_size, back_size, val);
    (*op).BindInput(inputs[0]).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

using AddMapper =
    SimpleOpWithFusedActiovationMapper<tim::vx::ops::Add, TfLiteAddParams>;
using SubMapper =
    SimpleOpWithFusedActiovationMapper<tim::vx::ops::Sub, TfLiteSubParams>;
using DivMapper =
    SimpleOpWithFusedActiovationMapper<tim::vx::ops::Div, TfLiteDivParams>;
using MulMapper =
    SimpleOpWithFusedActiovationMapper<tim::vx::ops::Multiply, TfLiteMulParams>;

template <tim::vx::ResizeType resizeType>
struct ResizeMapper
    : public OpMapperBase<TfLiteResizeNearestNeighborParams,
                          TransposeInputAction<0, 1, 2, 0, 3>,
                          TransposeOutputAction<0, 2, 0, 1, 3>> {
  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    LOG(INFO) << "Check Resize(" << static_cast<int>(resizeType) << ")";
    if (resizeType == tim::vx::ResizeType::BILINEAR) {
      int input_index = node->inputs->data[0];
      if (context->tensors[input_index].type == kTfLiteInt8 &&
          context->tensors[input_index].quantization.type ==
              kTfLiteNoQuantization) {
        LOG(ERROR) << "Int8 input without quantization is not supported in "
                      "ResizeBilinear";
        return false;
      }
    }
    int size_tensor_idx = node->inputs->data[1];
    const uint8_t* tensor_data = reinterpret_cast<const uint8_t*>(
        context->tensors[size_tensor_idx].data.raw_const);
    return tensor_data != nullptr;
  }

  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    LOG(INFO) << "Creating Resize(" << static_cast<int>(resizeType) << ") op";
    const auto builtin =
        reinterpret_cast<const TfLiteResizeNearestNeighborParams*>(params);
    auto size_tensor = inputs[1];

    std::vector<int> size(size_tensor->GetShape()[0]);
    size_tensor->CopyDataFromTensor(size.data());

    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Resize>(
        resizeType,
        0.0f,
        builtin->align_corners,
        builtin->half_pixel_centers,
        size[0],
        size[1]);

    (*op).BindInput(inputs[0]);
    (*op).BindOutput(outputs[0]);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct AddNMapper : public OpMapperBase<EmptyStructPlaceholder> {
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    LOG(INFO) << "Creating AddN op";
    auto output_tensor = outputs[0];
    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::AddN>(
        inputs.size());

    (*op).BindInputs(inputs);
    (*op).BindOutput(output_tensor);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct SplitMapper : public OpMapperBase<TfLiteSplitParams> {
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    LOG(INFO) << "Creating Split op";
    const auto builtin = reinterpret_cast<const TfLiteSplitParams*>(params);

    auto axis_tensor = inputs[0];
    auto input_tensor = inputs[1];

    int32_t axis = 0;
    axis_tensor->CopyDataFromTensor(&axis);
    axis =
        vx::delegate::utils::ConvertAxis(axis, input_tensor->GetShape().size());

    std::vector<uint32_t> slices;
    for (auto& o : outputs) {
      slices.push_back(o->GetShape()[axis]);
    }

    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Split>(
        axis, slices);

    (*op).BindInput(input_tensor);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct Space2DepthMapper
    : public OpMapperBase<TfLiteSpaceToDepthParams,
                          TransposeInputAction<0, 1, 2, 0, 3>,
                          TransposeOutputAction<0, 2, 0, 1, 3>> {
  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    for (int i = 0; i < node->inputs->size; i++) {
      int input_index = node->inputs->data[i];
      if (context->tensors[input_index].type == kTfLiteInt32) {
        LOG(ERROR) << "Int32 input is not supported in Space To Depth";
        return false;
      }
      if (context->tensors[input_index].type == kTfLiteInt64) {
        LOG(ERROR) << "Int64 input is not supported in Space To Depth";
        return false;
      }
    }
    return true;
  }

  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    LOG(INFO) << "Create SpaceToDepth op";
    const auto builtin =
        reinterpret_cast<const TfLiteSpaceToDepthParams*>(params);

    std::vector<int> block({builtin->block_size, builtin->block_size});
    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::SpaceToDepth>(
        block);
    (*op).BindInput(inputs[0]);
    (*op).BindOutput(outputs[0]);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct Depth2SpaceMapper
    : public OpMapperBase<TfLiteDepthToSpaceParams,
                          TransposeInputAction<0, 1, 2, 0, 3>,
                          TransposeOutputAction<0, 2, 0, 1, 3>> {
  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    for (int i = 0; i < node->inputs->size; i++) {
      int input_index = node->inputs->data[i];
      if (context->tensors[input_index].type == kTfLiteInt32) {
        LOG(INFO) << "Int32 input is not supported in Space To Depth";
        return false;
      }
      if (context->tensors[input_index].type == kTfLiteInt64) {
        LOG(INFO) << "Int64 input is not supported in Space To Depth";
        return false;
      }
    }
    return true;
  }

  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    LOG(INFO) << "Create DepthToSpace op";
    const auto builtin =
        reinterpret_cast<const TfLiteDepthToSpaceParams*>(params);

    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::DepthToSpace>(
        builtin->block_size);

    (*op).BindInput(inputs[0]);
    (*op).BindOutput(outputs[0]);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct PreluMapper : public OpMapperBase<EmptyStructPlaceholder> {
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    LOG(INFO) << "Create Prelu op";
    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Prelu>(0);

    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct Transpose : public OpMapperBase<TfLiteTransposeParams> {
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    LOG(INFO) << "Create Transpose op";
    auto perm_tensor = inputs[1];
    std::vector<uint32_t> perm(perm_tensor->GetShape()[0]);
    perm_tensor->CopyDataFromTensor(perm.data());
    std::vector<uint32_t> ovx_perm =
        vx::delegate::utils::GetOvxTransposePerm(perm);
    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Transpose>(
        ovx_perm);

    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct Gather : public OpMapperBase<TfLiteGatherParams> {
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    LOG(INFO) << "Create Gather op";
    const auto builtin = reinterpret_cast<const TfLiteGatherParams*>(params);
    int axis = vx::delegate::utils::ConvertAxis(builtin->axis,
                                                inputs[0]->GetShape().size());
    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Gather>(axis);

    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct GatherNd : public OpMapperBase<EmptyStructPlaceholder> {
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    LOG(INFO) << "Create GatherNd op";
    auto axis = std::make_shared<int32_t>(0);
    inputs[1] = ReverseInputTensor(delegate, inputs[1], axis.get(), 1);
    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::GatherNd>();

    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct Batch2Space : public OpMapperBase<TfLiteBatchToSpaceNDParams,
                                         TransposeInputAction<0, 1, 2, 0, 3>,
                                         TransposeOutputAction<0, 2, 0, 1, 3>> {
  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    int input_index = node->inputs->data[0];
    if (context->tensors[input_index].dims->size != 4) {
      LOG(ERROR) << "batch2space in vx-delagate only support 4D input";
      return false;
    }
    int block_index = node->inputs->data[1];
    if (context->tensors[block_index].dims->data[0] != 2) {
      LOG(ERROR) << "batch2space in vx-delagate only support the input whose "
                    "spatial dimensions is 2";
      return false;
    }
    return true;
  }

  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    LOG(INFO) << "Create Batch2Space op";
    // the value of block_size_num should be 2.
    int block_size_num = inputs[1]->GetShape()[0];
    std::vector<int> block_size(block_size_num);
    std::vector<int> crop(block_size_num * 2);
    inputs[1]->CopyDataFromTensor(block_size.data());
    inputs[2]->CopyDataFromTensor(crop.data());
    block_size = std::vector<int>(block_size.rbegin(), block_size.rend());
    std::vector<int> new_crop =
        vx::delegate::utils::TransposeVec<int>(crop, {2, 3, 0, 1});
    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Batch2Space>(
        block_size, new_crop);

    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct Space2Batch : public OpMapperBase<TfLiteSpaceToBatchNDParams,
                                         TransposeInputAction<0, 1, 2, 0, 3>,
                                         TransposeOutputAction<0, 2, 0, 1, 3>> {
  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    int input_index = node->inputs->data[0];
    if (context->tensors[input_index].dims->size != 4) {
      LOG(ERROR) << "space2batch in vx-delegate only support 4D input";
      return false;
    }
    int block_index = node->inputs->data[1];
    if (context->tensors[block_index].dims->data[0] != 2) {
      LOG(ERROR) << "space2batch in vx-delegate only support the input whose "
                    "spatial dimensions is 2";
      return false;
    }
    return true;
  }
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    LOG(INFO) << "Create Space2Batch op";
    // the value of block_size_num should be 2.
    int block_size_num = inputs[1]->GetShape()[0];
    std::vector<int> block_size(block_size_num);
    std::vector<int> pad(block_size_num * 2);
    inputs[1]->CopyDataFromTensor(block_size.data());
    inputs[2]->CopyDataFromTensor(pad.data());
    block_size = std::vector<int>(block_size.rbegin(), block_size.rend());
    std::vector<int> new_pad =
        vx::delegate::utils::TransposeVec<int>(pad, {2, 3, 0, 1});
    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Space2Batch>(
        block_size, new_pad);
    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct CustomOpMap : public OpMapperBase<EmptyStructPlaceholder> {
  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    return false;
  }
};

template <typename T_OperationType>
struct ReduceOpMapper : public OpMapperBase<TfLiteReducerParams> {
  std::string name_;

  ReduceOpMapper(std::string name) : name_(name) {}
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    LOG(INFO) << "Create reduce" << name_ << "op";
    const auto builtin = reinterpret_cast<const TfLiteReducerParams*>(params);
    auto keep_dims = builtin->keep_dims;

    uint32_t axis_num = inputs[1]->GetShape()[0];
    std::vector<int32_t> axis(axis_num);
    inputs[1]->CopyDataFromTensor(axis.data());
    for (uint32_t i = 0; i < axis_num; i++) {
      axis[i] = vx::delegate::utils::ConvertAxis(axis[i],
                                                 inputs[0]->GetShape().size());
    }
    auto op =
        delegate->GetGraph()->CreateOperation<T_OperationType>(axis, keep_dims);

    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct ExpandDimsMapper : public OpMapperBase<EmptyStructPlaceholder> {
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    LOG(INFO) << "Create ExpandDims op";

    auto input_shape = inputs[0]->GetShape();
    int axis = 0;
    inputs[1]->CopyDataFromTensor(&axis);
    auto output_shape = outputs[0]->GetShape();
    uint32_t new_axis =
        vx::delegate::utils::ConvertAxis(axis, output_shape.size());

    std::vector<uint32_t> expanded_shape(input_shape);
    expanded_shape.insert(expanded_shape.begin() + new_axis, 1);

    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Reshape>(
        expanded_shape);

    (*op).BindInput(inputs[0]);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct LeakyReluMapper : public OpMapperBase<TfLiteLeakyReluParams> {
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    LOG(INFO) << "Create LeakyRelu op";
    const auto builtin = reinterpret_cast<const TfLiteLeakyReluParams*>(params);
    auto alpha = builtin->alpha;
    auto op =
        delegate->GetGraph()->CreateOperation<tim::vx::ops::LeakyRelu>(alpha);

    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct Slice : public OpMapperBase<EmptyStructPlaceholder> {
  bool IsOpSupported(TfLiteContext* context,
                     TfLiteNode* node,
                     const TfLiteRegistration* registration) const override {
    int input_index = node->inputs->data[0];
    int output_index = node->outputs->data[0];
    int input_dim_size = context->tensors[input_index].dims->size;
    int batch_in = context->tensors[input_index].dims->data[0];
    int batch_out = context->tensors[output_index].dims->data[0];

    if (input_dim_size > 3 && (batch_in != batch_out)) {
      LOG(ERROR) << "vx-delegate doesn't support slice in batch.";
      return false;
    }
  }

  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    LOG(INFO) << "Create Slice op";
    auto input_tensor = inputs[0];
    auto begin_tensor = inputs[1];
    auto size_tensor = inputs[2];
    uint32_t input_dims = input_tensor->GetShape().size();
    uint32_t begin_size = begin_tensor->GetShape()[0];
    uint32_t size_size = size_tensor->GetShape()[0];
    std::vector<int32_t> begin(begin_size);
    std::vector<int32_t> size(size_size);
    begin_tensor->CopyDataFromTensor(begin.data());
    size_tensor->CopyDataFromTensor(size.data());

    std::reverse(begin.begin(), begin.end());
    std::reverse(size.begin(), size.end());

    for (int i = 0; i < size.size(); i++) {
      if (size[i] == -1) { // If size[i] == -1, that means extract all elements
                           // of demension i.
        size[i] = input_tensor->GetShape()[i];
      }
    }
    
    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Slice>(
        input_dims, begin, size);

    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

template <typename T_OperationType>
struct LogicalOpMapper : public OpMapperBase<EmptyStructPlaceholder> {
  std::string name_;

  LogicalOpMapper(std::string name) : name_(name) {}
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    LOG(INFO) << "Creating Logical" << name_ << " op";

    auto op = delegate->GetGraph()->CreateOperation<T_OperationType>();
    (*op).BindInputs(inputs).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

using createIOpMapItemFunc = std::function<std::unique_ptr<IOpMapper>()>;
static const std::map<int, createIOpMapItemFunc> reg = {
#define REGISTER_OP_MAPPER(TFLITE_OP_CODE, MAPPER_TYPE, ...)                  \
  {                                                                           \
    TFLITE_OP_CODE, [] { return std::make_unique<MAPPER_TYPE>(__VA_ARGS__); } \
  }

    REGISTER_OP_MAPPER(kTfLiteBuiltinFullyConnected, FullyConnectedMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinSoftmax, SoftmaxMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinConv2d, Conv2dMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinMaxPool2d,
                       Pool2dMapper<tim::vx::PoolType::MAX>),
    REGISTER_OP_MAPPER(kTfLiteBuiltinAveragePool2d,
                       Pool2dMapper<tim::vx::PoolType::AVG_ANDROID>),
    REGISTER_OP_MAPPER(kTfLiteBuiltinDepthwiseConv2d, DepthwiseConv2dMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinDequantize,
                       SimpleOpMapper<tim::vx::ops::DataConvert>,
                       "Dequantize"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinQuantize,
                       SimpleOpMapper<tim::vx::ops::DataConvert>,
                       "Quantize"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinConcatenation, ConcatenationMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinLocalResponseNormalization,
                       LocalResponseNormalizationMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinL2Normalization, L2NormalizationMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinReshape, ReshapeMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinStridedSlice, StridedSliceMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinPad, PadMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinExpandDims, ExpandDimsMapper),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinAbs, SimpleOpMapper<tim::vx::ops::Abs>, "Abs"),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinSin, SimpleOpMapper<tim::vx::ops::Sin>, "Sin"),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinExp, SimpleOpMapper<tim::vx::ops::Exp>, "Exp"),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinLog, SimpleOpMapper<tim::vx::ops::Log>, "Log"),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinSqrt, SimpleOpMapper<tim::vx::ops::Sqrt>, "Sqrt"),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinRsqrt, SimpleOpMapper<tim::vx::ops::Rsqrt>, "Rsqrt"),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinSquare, SimpleOpMapper<tim::vx::ops::Square>, "Square"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinLogicalNot,
                       SimpleOpMapper<tim::vx::ops::LogicalNot>,
                       "LogicalNot"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinHardSwish,
                       SimpleOpMapper<tim::vx::ops::HardSwish>,
                       "HardSwish"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinMinimum,
                       SimpleOpMapper<tim::vx::ops::Minimum>,
                       "Minimum"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinMaximum,
                       SimpleOpMapper<tim::vx::ops::Maximum>,
                       "Maximum"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinAdd, AddMapper, "Add"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinSub, SubMapper, "Sub"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinDiv, DivMapper, "Div"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinMul, MulMapper, "Multiply"),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinPow, SimpleOpMapper<tim::vx::ops::Pow>, "Pow"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinResizeNearestNeighbor,
                       ResizeMapper<tim::vx::ResizeType::NEAREST_NEIGHBOR>),
    REGISTER_OP_MAPPER(kTfLiteBuiltinResizeBilinear,
                       ResizeMapper<tim::vx::ResizeType::BILINEAR>),
    REGISTER_OP_MAPPER(kTfLiteBuiltinAddN, AddNMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinSplit, SplitMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinSpaceToDepth, Space2DepthMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinDepthToSpace, Depth2SpaceMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinPrelu, PreluMapper),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinElu, SimpleOpMapper<tim::vx::ops::Elu>, "Elu"),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinRelu, SimpleOpMapper<tim::vx::ops::Relu>, "Relu"),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinReluN1To1, SimpleOpMapper<tim::vx::ops::Relu1>, "Relu1"),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinRelu6, SimpleOpMapper<tim::vx::ops::Relu6>, "Relu6"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinTranspose, Transpose),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinNeg, SimpleOpMapper<tim::vx::ops::Neg>, "Neg"),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinTanh, SimpleOpMapper<tim::vx::ops::Tanh>, "tanh"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinGather, Gather),
    REGISTER_OP_MAPPER(kTfLiteBuiltinGatherNd, GatherNd),
    REGISTER_OP_MAPPER(kTfLiteBuiltinBatchToSpaceNd, Batch2Space),
    REGISTER_OP_MAPPER(kTfLiteBuiltinSpaceToBatchNd, Space2Batch),
    REGISTER_OP_MAPPER(kTfLiteBuiltinReduceMin,
                       ReduceOpMapper<tim::vx::ops::ReduceMin>,
                       "Min"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinReduceMax,
                       ReduceOpMapper<tim::vx::ops::ReduceMax>,
                       "Max"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinReduceAny,
                       ReduceOpMapper<tim::vx::ops::ReduceAny>,
                       "Any"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinReduceProd,
                       ReduceOpMapper<tim::vx::ops::ReduceProd>,
                       "Prod"),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinMean, ReduceOpMapper<tim::vx::ops::ReduceMean>, "Mean"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinLeakyRelu, LeakyReluMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinLogicalAnd,
                       LogicalOpMapper<tim::vx::ops::LogicalAnd>,
                       "And"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinLogicalOr,
                       LogicalOpMapper<tim::vx::ops::LogicalOr>,
                       "Or"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinSlice, Slice),

#undef REGISTER_OP_MAPPTER
};

static const std::map<std::string, createIOpMapItemFunc> custom_reg = {
#define REGISTER_CUSTOM_OP(CUSTOM_NAME, MAPPER_TYPE, ...)                  \
  {                                                                        \
    CUSTOM_NAME, [] { return std::make_unique<MAPPER_TYPE>(__VA_ARGS__); } \
  }

    REGISTER_CUSTOM_OP("WRNN_BIDI_SEQGRU", CustomOpMap),

#undef REGISTER_CUSTOM_OP
};

template <typename T>
struct OperationMapConstructor {
  T supported_builtins;
  OperationMapConstructor(
      const std::map<typename T::key_type, createIOpMapItemFunc> reg) {
    LOG(INFO) << "Initialize supported_builtins";
    for (const auto& kv : reg) {
      supported_builtins.insert(std::make_pair(kv.first, kv.second()));
    }
  }
};

const OperationMapItemType& SupportedBuiltinOps() {
  static OperationMapConstructor<OperationMapItemType> c(reg);

  return c.supported_builtins;
}

const CustomOperationMapItemType& SupportedBuiltinCustomOps() {
  static OperationMapConstructor<CustomOperationMapItemType> c(custom_reg);

  return c.supported_builtins;
}

}  // namespace op_map
}  // namespace vx