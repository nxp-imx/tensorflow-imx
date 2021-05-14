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

#ifndef TENSORFLOW_LITE_DELEGATES_VX_DELEGAGE_DELEGATE_MAIN_H
#define TENSORFLOW_LITE_DELEGATES_VX_DELEGAGE_DELEGATE_MAIN_H

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/context.h"
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/operation.h"
#include "tim/vx/tensor.h"
#include "tim/lite/execution.h"
#include "tim/lite/handle.h"

namespace vx {
namespace delegate {

TfLiteDelegate* VxDelegate();

class Delegate;

struct OpData {
  std::vector<int> subgraph_inputs;
  std::vector<int> subgraph_outputs;
  std::vector<int> subgraph_states;

  std::unique_ptr<Delegate> delegate;
};

class Delegate {
 public:
  static TfLiteDelegate* Create();
  static bool SupportedOp(TfLiteContext* context,
                          TfLiteNode* node,
                          const TfLiteRegistration* registration);

  Delegate();
  ~Delegate() {}

  virtual std::unique_ptr<OpData> Init(TfLiteContext* context,
                               const TfLiteDelegateParams* params);
  virtual TfLiteStatus Prepare(const OpData& op_data,
                       TfLiteContext* context,
                       TfLiteNode* node);
  virtual TfLiteStatus Invoke(const OpData& op_data,
                      TfLiteContext* context,
                      TfLiteNode* node);
  std::vector<std::shared_ptr<tim::vx::Operation>>& GetOps() { return ops_; }
  std::shared_ptr<tim::vx::Graph>& GetGraph() { return graph_; }
  std::vector<std::shared_ptr<tim::vx::Tensor>>& GetTensors() {
    return tensors_;
  }

 protected:
  virtual bool IsCompiled() const { return compiled_; }
  virtual bool Compile(const OpData& op_data, TfLiteContext* context);

 private:
  struct OperationDataType {
    int builtin_code;
    std::string custom_name;
    std::vector<int> inputs;
    std::vector<int> outputs;
    std::vector<int> states;
    std::vector<uint8_t> builtin_data;
  };

  std::shared_ptr<tim::vx::Context> context_;
  std::shared_ptr<tim::vx::Graph> graph_;
  std::vector<std::shared_ptr<tim::vx::Tensor>> tensors_;
  std::vector<std::shared_ptr<tim::vx::Tensor>> state_tensors_;
  std::vector<std::shared_ptr<tim::vx::Operation>> ops_;
  std::vector<OperationDataType> operations_;
  bool compiled_;
};

class LiteBuffer {
  public:
   LiteBuffer(size_t bytes, uint32_t align_size = 64);
   ~LiteBuffer();
   void* data() { return data_; }
   size_t bytes() { return bytes_; }
  private:
   void* data_;
   size_t bytes_;
};

class LiteDelegate : public Delegate {
 public:
  std::unique_ptr<OpData> Init(TfLiteContext* context,
                               const TfLiteDelegateParams* params) override;
  TfLiteStatus Invoke(const OpData& op_data,
                      TfLiteContext* context,
                      TfLiteNode* node) override;

 protected:
  bool IsCompiled() const override { return Delegate::IsCompiled() && compiled_; }
  bool Compile(const OpData& op_data, TfLiteContext* context) override;

 private:
  std::shared_ptr<tim::lite::Execution> exec_;
  std::vector<std::shared_ptr<LiteBuffer>> inputs_;
  std::vector<std::shared_ptr<LiteBuffer>> outputs_;
  bool compiled_;
};

}  // namespace delegate
}  // namespace vx

#endif /* TENSORFLOW_LITE_DELEGATES_VX_DELEGAGE_DELEGATE_MAIN_H */
