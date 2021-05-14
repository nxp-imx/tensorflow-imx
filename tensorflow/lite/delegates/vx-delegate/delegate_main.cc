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

#include "delegate_main.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "op_map.h"
#include "utils.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace {

TfLiteRegistration DelegateNodeRegistration() {
  TfLiteRegistration r;

  r.builtin_code = kTfLiteBuiltinDelegate;
  r.custom_name = "Vx Delegate";

  r.init = [](TfLiteContext* context, const char* buffer, size_t) -> void* {
    auto* params = reinterpret_cast<const TfLiteDelegateParams*>(buffer);
    std::unique_ptr<vx::delegate::Delegate> delegate(
        new vx::delegate::Delegate);
    //std::unique_ptr<vx::delegate::Delegate> delegate(
    //    new vx::delegate::LiteDelegate);

    std::unique_ptr<vx::delegate::OpData> op_data =
        delegate->Init(context, params);
    op_data->delegate.swap(delegate);
    return op_data.release();
  };

  r.free = [](TfLiteContext* context, void* buffer) -> void {
    std::unique_ptr<vx::delegate::OpData> op_data(
        reinterpret_cast<vx::delegate::OpData*>(buffer));
    op_data->delegate = nullptr;
  };

  r.prepare = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
    auto op_data = reinterpret_cast<vx::delegate::OpData*>(node->user_data);
    return op_data->delegate->Prepare(*op_data, context, node);
  };

  r.invoke = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
    auto op_data = reinterpret_cast<vx::delegate::OpData*>(node->user_data);
    return op_data->delegate->Invoke(*op_data, context, node);
  };

  return r;
}

TfLiteStatus PrepareDelegate(TfLiteContext* context, TfLiteDelegate* delegate) {
  TfLiteIntArray* plan;
  TfLiteNode* node;
  TfLiteRegistration* registration;
  TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &plan));

  // Get a list of supported nodes.
  std::vector<int> supported_nodes = {0};
  for (int node_index : tflite::TfLiteIntArrayView(plan)) {
    TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
        context, node_index, &node, &registration));
    if (vx::delegate::Delegate::SupportedOp(context, node, registration)) {
      supported_nodes.push_back(node_index);
    }
  }

  // Set first element to the number of nodes to replace.
  supported_nodes[0] = supported_nodes.size() - 1;

  // Replace supported subgraphs.
  return context->ReplaceNodeSubsetsWithDelegateKernels(
      context,
      DelegateNodeRegistration(),
      reinterpret_cast<TfLiteIntArray*>(supported_nodes.data()),
      delegate);
}

TfLiteStatus CopyFromBufferHandle(TfLiteContext* context,
                                  TfLiteDelegate* delegate,
                                  TfLiteBufferHandle buffer_handle,
                                  TfLiteTensor* tensor) {
  // Copies the data from delegate buffer into the tensor raw memory.
  LOG(INFO) << "CopyFromBufferHandle handle:" << buffer_handle
            << " tensor:" << tensor->name;
  return kTfLiteOk;
}

void FreeBufferHandle(TfLiteContext* context,
                      TfLiteDelegate* delegate,
                      TfLiteBufferHandle* handle) {
  // Do any cleanups.
  LOG(INFO) << "FreeBufferHandle handle:" << *handle;
}

std::vector<uint32_t> TfLiteTensorDims(const TfLiteTensor* tensor) {
  std::vector<uint32_t> dims(tensor->dims->size);
  for (std::vector<uint32_t>::size_type i = 0; i < dims.size(); i++) {
    dims[i] = tensor->dims->data[i];
  }
  return dims;
}

tim::vx::DataType TfLiteDtypeToVsiDtype(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
      return tim::vx::DataType::FLOAT32;
    case kTfLiteInt32:
      return tim::vx::DataType::INT32;
    case kTfLiteUInt8:
      return tim::vx::DataType::UINT8;
    case kTfLiteInt16:
      return tim::vx::DataType::INT16;
    case kTfLiteInt8:
      return tim::vx::DataType::INT8;
    case kTfLiteBool:
      return tim::vx::DataType::INT8;
    case kTfLiteFloat16:
      return tim::vx::DataType::FLOAT16;
    default:
      LOG(ERROR) << "Unsuppoted type:" << type;
      break;
  }

  return tim::vx::DataType::FLOAT32;
}

bool IsConstTensor(const TfLiteTensor* tensor) {
  const uint8_t* tensor_data =
      reinterpret_cast<const uint8_t*>(tensor->data.raw_const);
  return tensor_data != nullptr;
}

bool IsVariableTensor(const TfLiteTensor* tensor) {
  return tensor->is_variable;
}

tim::vx::TensorSpec CreateTensorSpec(
    const TfLiteTensor* tensor,
    const std::vector<uint32_t>& perm,
    tim::vx::TensorAttribute attr = tim::vx::TensorAttribute::TRANSIENT) {
  tim::vx::DataType datatype = TfLiteDtypeToVsiDtype(tensor->type);
  std::vector<uint32_t> dims(TfLiteTensorDims(tensor));
  tim::vx::ShapeType whcn_shape(dims.size());

  if (dims.size() == 0) {
    // Use rank 1, shape {1} operand for TFLite scalar tensors.
    dims.push_back(1);
  }

  if (perm.size() > 0) {
    assert(perm.size() == dims.size());
    for (size_t i = 0; i < perm.size(); i++) {
      whcn_shape[i] = dims[perm[i]];
    }
    std::reverse(whcn_shape.begin(), whcn_shape.end());
  } else {
    whcn_shape.assign(dims.rbegin(), dims.rend());
  }

  if (tensor->quantization.type == kTfLiteAffineQuantization) {
    const TfLiteAffineQuantization* params =
        reinterpret_cast<const TfLiteAffineQuantization*>(
            tensor->quantization.params);
    tim::vx::Quantization quantization;
    std::vector<float> scales(params->scale->data,
                              params->scale->data + params->scale->size);
    std::vector<int32_t> zero_points(
        params->zero_point->data,
        params->zero_point->data + params->zero_point->size);

    tim::vx::QuantType qtype = tim::vx::QuantType::ASYMMETRIC;
    if (scales.size() > 1) {
      qtype = tim::vx::QuantType::SYMMETRIC_PER_CHANNEL;
      int32_t channel_dim = perm.size() > 1
                                ? vx::delegate::utils::TransposeChannelDim(
                                      perm, params->quantized_dimension)
                                : params->quantized_dimension;
      int32_t vx_channel_dim =
          vx::delegate::utils::ConvertAxis(channel_dim, dims.size());
      quantization =
          tim::vx::Quantization(qtype, vx_channel_dim, scales, zero_points);
    } else {
      quantization = tim::vx::Quantization(qtype, scales[0], zero_points[0]);
    }

    return tim::vx::TensorSpec(datatype, whcn_shape, attr, quantization);
  }

  return tim::vx::TensorSpec(datatype, whcn_shape, attr);
}

bool TransposeTensorData(const TfLiteTensor* tensor,
                         const std::vector<uint32_t>& perm,
                         std::vector<uint8_t>& data_out) {
  const uint8_t* tensor_data =
      reinterpret_cast<const uint8_t*>(tensor->data.raw_const);
  if (!tensor_data) {
    return false;
  }

  tflite::TransposeParams params;
  std::vector<int32_t> output_shape;

  params.perm_count = perm.size();
  output_shape.resize(perm.size());
  for (size_t i = 0; i < perm.size(); i++) {
    params.perm[i] = perm[i];
    output_shape[i] = tensor->dims->data[perm[i]];
  }
  data_out.resize(tensor->bytes);
  switch (tensor->type) {
    case kTfLiteFloat32:
    case kTfLiteInt32:
      tflite::reference_ops::Transpose(
          params,
          tflite::GetTensorShape(tensor),
          tflite::GetTensorData<int32_t>(tensor),
          tflite::RuntimeShape(
              static_cast<int>(output_shape.size()),
              reinterpret_cast<const int32_t*>(output_shape.data())),
          reinterpret_cast<int32_t*>(data_out.data()));
      break;
    case kTfLiteInt16:
    case kTfLiteFloat16:
      tflite::reference_ops::Transpose(
          params,
          tflite::GetTensorShape(tensor),
          tflite::GetTensorData<int16_t>(tensor),
          tflite::RuntimeShape(
              static_cast<int>(output_shape.size()),
              reinterpret_cast<const int32_t*>(output_shape.data())),
          reinterpret_cast<int16_t*>(data_out.data()));
      break;
    case kTfLiteUInt8:
    case kTfLiteInt8:
      tflite::reference_ops::Transpose(
          params,
          tflite::GetTensorShape(tensor),
          tflite::GetTensorData<int8_t>(tensor),
          tflite::RuntimeShape(
              static_cast<int>(output_shape.size()),
              reinterpret_cast<const int32_t*>(output_shape.data())),
          reinterpret_cast<int8_t*>(data_out.data()));
      break;
    default:
      LOG(ERROR) << "Unsupported type: " << tensor->type;
      return false;
  }

  return true;
}

std::shared_ptr<tim::vx::Tensor> CreateTensor(
    std::shared_ptr<tim::vx::Graph>& graph,
    const TfLiteTensor* tensor,
    const tim::vx::TensorAttribute& attr,
    const std::vector<uint32_t>& perm) {
  const uint8_t* tensor_data = nullptr;
  tim::vx::TensorSpec spec = CreateTensorSpec(tensor, perm, attr);
  switch (attr) {
    case tim::vx::TensorAttribute::INPUT:
    case tim::vx::TensorAttribute::OUTPUT:
    case tim::vx::TensorAttribute::VARIABLE:
      break;
    case tim::vx::TensorAttribute::CONSTANT:
      tensor_data = reinterpret_cast<const uint8_t*>(tensor->data.raw_const);
      if (perm.size() > 0) {
        std::vector<uint8_t> data_transposed;
        if (TransposeTensorData(tensor, perm, data_transposed)) {
          return graph->CreateTensor(
              spec, reinterpret_cast<const void*>(data_transposed.data()));
        }
      }
      break;
    case tim::vx::TensorAttribute::TRANSIENT:
      break;
    default:
      break;
  }
  return graph->CreateTensor(spec, reinterpret_cast<const void*>(tensor_data));
}

std::vector<std::shared_ptr<tim::vx::Tensor>> MapIndexesToTensors(
    const std::vector<std::shared_ptr<tim::vx::Tensor>>& tensors,
    const std::vector<int>& indexes) {
  std::vector<std::shared_ptr<tim::vx::Tensor>> out_tensors;
  std::transform(indexes.begin(),
                 indexes.end(),
                 back_inserter(out_tensors),
                 [tensors](const int& index) {
                   return tensors[(index + tensors.size()) % tensors.size()];
                 });
  return out_tensors;
}

void GetTransposePerm(int32_t op,
                      size_t port_idx,
                      const TfLiteTensor* tensor,
                      std::vector<uint32_t>& perm) {
  std::vector<uint32_t> dims(TfLiteTensorDims(tensor));
  switch (op) {
    case kTfLiteBuiltinConv2d:
    case kTfLiteBuiltinDepthwiseConv2d:
    case kTfLiteBuiltinTransposeConv:
      // filter
      if (port_idx == 1) {
        perm.resize(dims.size());
        perm[0] = 0;
        perm[1] = 3;
        perm[2] = 1;
        perm[3] = 2;
      }
      break;

    default:
      break;
  }
}

}  // namespace

namespace vx {
namespace delegate {

TfLiteDelegate* VxDelegate() {
  static TfLiteDelegate* delegate = vx::delegate::Delegate::Create();
  return delegate;
}

bool Delegate::SupportedOp(TfLiteContext* context,
                           TfLiteNode* node,
                           const TfLiteRegistration* registration) {
  if (registration->custom_name != nullptr) {
    const auto& supported_custom_ops = vx::op_map::SupportedBuiltinCustomOps();
    const auto& it = supported_custom_ops.find(registration->custom_name);
    if (supported_custom_ops.end() != it) {
      return it->second->IsSupported(context, node, registration);
    }
  }

  const auto& supported_builtins = vx::op_map::SupportedBuiltinOps();
  const auto& it = supported_builtins.find(
      static_cast<TfLiteBuiltinOperator>(registration->builtin_code));
  if (supported_builtins.end() != it) {
    return it->second->IsSupported(context, node, registration);
  }

  LOG(ERROR) << "Fallback unsupported op " << registration->builtin_code
             << " to TfLite";

  return false;
}

TfLiteDelegate* Delegate::Create() {
  TfLiteDelegate* delegate = new TfLiteDelegate();

  std::memset(delegate, 0, sizeof(TfLiteDelegate));
  delegate->flags = kTfLiteDelegateFlagsNone;
  delegate->Prepare = &PrepareDelegate;
  delegate->CopyFromBufferHandle = &CopyFromBufferHandle;
  delegate->FreeBufferHandle = &FreeBufferHandle;

  return delegate;
}

std::unique_ptr<vx::delegate::OpData> Delegate::Init(
    TfLiteContext* context, const TfLiteDelegateParams* params) {
  LOG(INFO) << "vx_delegate Delegate::Init";

  compiled_ = false;
  tensors_.resize(context->tensors_size + 1 /* for placeholder*/);
  state_tensors_.resize(context->tensors_size + 1 /* for placeholder*/);

  std::unique_ptr<vx::delegate::OpData> op_data(new OpData());
  // Get the list of input and output tensors. This isn't for a single op, it's
  // for a subgraph.
  tflite::TfLiteIntArrayView input_tensors(params->input_tensors);
  for (int input_tensor_idx : input_tensors) {
    const auto& tensor = context->tensors[input_tensor_idx];
    if (tensor.allocation_type != kTfLiteMmapRo) {
      op_data->subgraph_inputs.push_back(input_tensor_idx);
    }
  }

  tflite::TfLiteIntArrayView output_tensors(params->output_tensors);
  std::copy(output_tensors.begin(),
            output_tensors.end(),
            std::back_inserter(op_data->subgraph_outputs));

  const auto& supported_customs = vx::op_map::SupportedBuiltinCustomOps();
  const auto& supported_builtins = vx::op_map::SupportedBuiltinOps();
  operations_.resize(params->nodes_to_replace->size);
  for (int i = 0; i < params->nodes_to_replace->size; i++) {
    TfLiteNode* node;
    TfLiteRegistration* reg;
    int node_idx = params->nodes_to_replace->data[i];
    context->GetNodeAndRegistration(context, node_idx, &node, &reg);
    tflite::TfLiteIntArrayView inputs(node->inputs);
    tflite::TfLiteIntArrayView outputs(node->outputs);

    auto& operation = operations_[i];

    if(reg->custom_name){
      operation.custom_name = reg->custom_name;
    }
    operation.builtin_code = reg->builtin_code;
    bool isbuiltinOp = operation.custom_name.empty();
    std::copy(
        inputs.begin(), inputs.end(), std::back_inserter(operation.inputs));
    std::copy(
        outputs.begin(), outputs.end(), std::back_inserter(operation.outputs));

    std::vector<int> states;
    if ((isbuiltinOp && supported_builtins.at(reg->builtin_code)
                ->GetStateTensorIndexes(context, node, reg, states) )|| (!isbuiltinOp &&
        supported_customs.at(operation.custom_name)
            ->GetStateTensorIndexes(context, node, reg, states))) {
      std::copy(
          states.begin(), states.end(), std::back_inserter(operation.states));

      // record state tensor index
      std::copy(states.begin(),
                states.end(),
                std::back_inserter(op_data->subgraph_states));
    }

    if(!isbuiltinOp && node->user_data) {
      operation.builtin_data.resize(supported_customs.at(operation.custom_name)->GetParamSize());
      memcpy(operation.builtin_data.data(),
             node->user_data,
             operation.builtin_data.size());
    }else if (isbuiltinOp && node->builtin_data) {
      operation.builtin_data.resize(supported_builtins.at(reg->builtin_code)->GetParamSize());
      memcpy(operation.builtin_data.data(),
             node->builtin_data,
             operation.builtin_data.size());
    }else{
      continue;
    }
  }

  return op_data;
}

TfLiteStatus Delegate::Prepare(const OpData& op_data,
                               TfLiteContext* context,
                               TfLiteNode* node) {
  LOG(INFO) << "Delegate::Prepare node:" << node->user_data;
  return kTfLiteOk;
}

TfLiteStatus Delegate::Invoke(const OpData& op_data,
                              TfLiteContext* context,
                              TfLiteNode* node) {
  LOG(INFO) << "Delegate::Invoke node:" << node->user_data;

  if (!Compile(op_data, context)) {
    return kTfLiteDelegateError;
  }
  // TODO(derekjchow): Return error if compilation failed.
  for (int tensor_idx : op_data.subgraph_inputs) {
    const TfLiteTensor& tf_tensor = context->tensors[tensor_idx];
    LOG(INFO) << "Copying input " << tensor_idx << ":" << tf_tensor.name;
    auto* tensor = tensors_[tensor_idx].get();
    if (!tensor) {
      LOG(FATAL) << "Failed to copy input tensor!";
    }

    const void* tensor_data =
        reinterpret_cast<const void*>(tf_tensor.data.raw_const);
    // TODO(derekjchow): Check result
    tensor->CopyDataToTensor(const_cast<void*>(tensor_data));
  }

  LOG(INFO) << "Invoking graph";
  if (!graph_->Run()) {
    LOG(FATAL) << "Failed to run graph";
  }

  for (int tensor_idx : op_data.subgraph_outputs) {
    TfLiteTensor& tf_tensor = context->tensors[tensor_idx];
    LOG(INFO) << "Copying output " << tensor_idx << ":" << tf_tensor.name;
    auto* tensor = tensors_[tensor_idx].get();
    if (!tensor) {
      LOG(FATAL) << "Failed to copy output tensor!";
    }

    void* tensor_data = reinterpret_cast<void*>(tf_tensor.data.raw);
    // TODO(derekjchow): Check result
    tensor->CopyDataFromTensor(tensor_data);
  }

  // Copy output states to input states
  for (int tensor_idx : op_data.subgraph_states) {
    TfLiteTensor& tf_tensor = context->tensors[tensor_idx];
    LOG(INFO) << "Copying state " << tensor_idx << ":" << tf_tensor.name;
    auto* tensor = state_tensors_[tensor_idx].get();
    if (!tensor) {
      LOG(FATAL) << "Disaster!";
    }

    void* tensor_data = reinterpret_cast<void*>(tf_tensor.data.raw);
    tensor->CopyDataFromTensor(tensor_data);
  }

  return kTfLiteOk;
}

bool Delegate::Compile(const OpData& op_data, TfLiteContext* context) {
  if (!compiled_) {
    // TODO(bo): Handling multi-thread use case
    context_ = tim::vx::Context::Create();
    graph_ = context_->CreateGraph();

    tensors_[tensors_.size() - 1] = graph_->CreateTensorPlaceHolder();

    // Create input tensors
    for (int tensor_idx : op_data.subgraph_inputs) {
      if (-1 != tensor_idx && tensors_[tensor_idx].get() == nullptr) {
        const auto tensor = &(context->tensors[tensor_idx]);
        tensors_[tensor_idx] =
            CreateTensor(graph_, tensor, tim::vx::TensorAttribute::INPUT, {});
      }
    }

    // Create output tensors
    for (int tensor_idx : op_data.subgraph_outputs) {
      if (-1 != tensor_idx && tensors_[tensor_idx].get() == nullptr) {
        const auto tensor = &(context->tensors[tensor_idx]);
        tensors_[tensor_idx] =
            CreateTensor(graph_, tensor, tim::vx::TensorAttribute::OUTPUT, {});
      }
    }

    // create op
    for (const auto& op_info : operations_) {
      auto& builtin_code = op_info.builtin_code;
      auto& custom_name = op_info.custom_name;
      auto& inputs = op_info.inputs;
      ;
      auto& outputs = op_info.outputs;
      auto& states = op_info.states;
      auto& builtin_data = op_info.builtin_data;

      std::vector<int> inputs_outputs;
      std::copy(
          inputs.begin(), inputs.end(), std::back_inserter(inputs_outputs));
      std::copy(
          outputs.begin(), outputs.end(), std::back_inserter(inputs_outputs));

      for (size_t port_idx = 0; port_idx < inputs_outputs.size(); port_idx++) {
        int tensor_idx = inputs_outputs[port_idx];
        if (-1 != tensor_idx && tensors_[tensor_idx].get() == nullptr) {
          std::vector<uint32_t> perm;
          auto tensor = &(context->tensors[tensor_idx]);
          tim::vx::TensorAttribute attr = tim::vx::TensorAttribute::TRANSIENT;
          if (IsConstTensor(tensor)) {
            attr = tim::vx::TensorAttribute::CONSTANT;
            GetTransposePerm(builtin_code, port_idx, tensor, perm);
          } else if (IsVariableTensor(tensor)) {
            attr = tim::vx::TensorAttribute::VARIABLE;
          } else {
            attr = tim::vx::TensorAttribute::TRANSIENT;
          }
          tensors_[tensor_idx] = CreateTensor(graph_, tensor, attr, perm);
        }
      }

      // create state output as graph output
      for (auto tensor_idx : states) {
        if (-1 != tensor_idx && state_tensors_[tensor_idx].get() == nullptr) {
          const auto tensor = &(context->tensors[tensor_idx]);
          state_tensors_[tensor_idx] = CreateTensor(
              graph_, tensor, tim::vx::TensorAttribute::OUTPUT, {});
        }
      }

      std::vector<std::shared_ptr<tim::vx::Tensor>> inputs_tensors =
          MapIndexesToTensors(tensors_, inputs);
      std::vector<std::shared_ptr<tim::vx::Tensor>> outputs_tensors =
          MapIndexesToTensors(tensors_, outputs);
      std::vector<std::shared_ptr<tim::vx::Tensor>> states_tensors =
          MapIndexesToTensors(state_tensors_, states);

      if (!custom_name.empty()) {
        vx::op_map::SupportedBuiltinCustomOps()
            .at(custom_name)
            ->MapOp(this,
                    inputs_tensors,
                    outputs_tensors,
                    states_tensors,
                    builtin_data.data());
      } else {
        vx::op_map::SupportedBuiltinOps()
            .at(builtin_code)
            ->MapOp(this,
                    inputs_tensors,
                    outputs_tensors,
                    states_tensors,
                    builtin_data.data());
      }
    }

    LOG(INFO) << "Verifying graph";
    compiled_ = graph_->Compile();
    if (!compiled_) {
      LOG(FATAL) << "Failed to verify graph";
      return false;
    }

    LOG(INFO) << "Verified graph";
  }
  return compiled_;
}

Delegate::Delegate() {}


LiteBuffer::LiteBuffer(size_t bytes, uint32_t align_size)
  : bytes_(bytes) {
#define MEM_ALIGN(x, align) (((x) + ((align)-1)) & ~((align)-1))
  uint32_t size = MEM_ALIGN(bytes, align_size);
  data_ = aligned_alloc(align_size, size);
#undef MEM_ALIGN
}

LiteBuffer::~LiteBuffer() {
  if (data_) {
    free(data_);
  }
}

std::unique_ptr<vx::delegate::OpData> LiteDelegate::Init(
    TfLiteContext* context, const TfLiteDelegateParams* params) {
  auto op_data = Delegate::Init(context, params);
  compiled_ = false;
  return op_data;
}

TfLiteStatus LiteDelegate::Invoke(const OpData& op_data,
                              TfLiteContext* context,
                              TfLiteNode* node) {
  if (!Compile(op_data, context)) {
    return kTfLiteDelegateError;
  }
  // TODO(derekjchow): Return error if compilation failed.
  for (size_t i = 0; i < op_data.subgraph_inputs.size(); i ++) {
    int tensor_idx = op_data.subgraph_inputs[i];
    const TfLiteTensor& tf_tensor = context->tensors[tensor_idx];
    LOG(INFO) << "Copying input " << tensor_idx << ":" << tf_tensor.name;
    auto buffer = inputs_[i];
    if (!buffer) {
      LOG(FATAL) << "Failed to copy input tensor!";
    }

    const void* tensor_data =
        reinterpret_cast<const void*>(tf_tensor.data.raw_const);
    // TODO(derekjchow): Check result
    memcpy(buffer->data(), const_cast<void*>(tensor_data), buffer->bytes());
  }

  LOG(INFO) << "Invoking execution";
  if (!exec_->Exec()) {
    LOG(FATAL) << "Failed to execution";
  }

  for (size_t i = 0; i < op_data.subgraph_outputs.size(); i ++) {
    int tensor_idx = op_data.subgraph_outputs[i];
    TfLiteTensor& tf_tensor = context->tensors[tensor_idx];
    LOG(INFO) << "Copying output " << tensor_idx << ":" << tf_tensor.name;
    auto buffer = outputs_[i];
    if (!buffer) {
      LOG(FATAL) << "Failed to copy output tensor!";
    }

    void* tensor_data = reinterpret_cast<void*>(tf_tensor.data.raw);
    // TODO(derekjchow): Check result
    memcpy(tensor_data, buffer->data(), buffer->bytes());
  }

  // TODO:
  // Copy output states to input states
  //for (int tensor_idx : op_data.subgraph_states) {
  //  TfLiteTensor& tf_tensor = context->tensors[tensor_idx];
  //  LOG(INFO) << "Copying state " << tensor_idx << ":" << tf_tensor.name;
  //  auto* tensor = state_tensors_[tensor_idx].get();
  //  if (!tensor) {
  //    LOG(FATAL) << "Disaster!";
  //  }

  //  void* tensor_data = reinterpret_cast<void*>(tf_tensor.data.raw);
  //  tensor->CopyDataFromTensor(tensor_data);
  //}
  return kTfLiteOk;
}

bool LiteDelegate::Compile(const OpData& op_data, TfLiteContext* context) {
  if (!compiled_) {
    if (Delegate::Compile(op_data, context)) {
      size_t size = 0;
      Delegate::GetGraph()->CompileToBinary(nullptr, &size);
      std::vector<uint8_t> binary_data;
      binary_data.resize(size);
      Delegate::GetGraph()->CompileToBinary(binary_data.data(), &size);
      exec_ = tim::lite::Execution::Create(binary_data.data(), binary_data.size());
      if (!exec_) {
        compiled_ = false;
        return kTfLiteDelegateError;
      }
      inputs_.clear();
      outputs_.clear();
      // Create input tensors
      std::vector<std::shared_ptr<tim::lite::Handle>> input_handles;
      for (size_t i = 0; i < op_data.subgraph_inputs.size(); i ++) {
        int tensor_idx = op_data.subgraph_inputs[i];
        std::shared_ptr<LiteBuffer> buffer;
        // Strip empty tensor for lite delegate
        if (-1 != tensor_idx) {
          const auto tensor = &(context->tensors[tensor_idx]);
          auto buffer = std::make_shared<LiteBuffer>(tensor->bytes);
          auto handle = exec_->RegisterHandle<tim::lite::UserHandle>(
            buffer->data(), buffer->bytes());
          input_handles.push_back(handle);
          inputs_.push_back(buffer);
        }
      }

      // Create output tensors
      std::vector<std::shared_ptr<tim::lite::Handle>> output_handles;
      for (size_t i = 0; i < op_data.subgraph_outputs.size(); i ++) {
        int tensor_idx = op_data.subgraph_outputs[i];
        // Strip empty tensor for lite delegate
        if (-1 != tensor_idx) {
          const auto tensor = &(context->tensors[tensor_idx]);
          auto buffer = std::make_shared<LiteBuffer>(tensor->bytes);
          auto handle = exec_->RegisterHandle<tim::lite::UserHandle>(
            buffer->data(), buffer->bytes());
          outputs_.push_back(buffer);
          output_handles.push_back(handle);
        }
      }
      exec_->BindInputs(input_handles).BindOutputs(output_handles);
      compiled_ = true;
    }
  }
  return compiled_;
}

}  // namespace delegate
}  // namespace vx
