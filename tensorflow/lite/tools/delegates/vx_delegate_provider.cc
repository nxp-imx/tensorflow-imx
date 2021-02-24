/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <string>

#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

namespace tflite {
namespace tools {

class VxDelegateProvider : public DelegateProvider {
 public:
  VxDelegateProvider() {
    default_params_.AddParam("use_vxdelegate", ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;

  std::string GetName() const final { return "VX Delegate"; }
};
REGISTER_DELEGATE_PROVIDER(VxDelegateProvider);

std::vector<Flag> VxDelegateProvider::CreateFlags(
    ToolParams* params) const {
  std::vector<Flag> flags = {
      CreateFlag<bool>("use_vxdelegate", params, "use VXdelegate")};
  return flags;
}

void VxDelegateProvider::LogParams(const ToolParams& params) const {
  TFLITE_LOG(INFO) << "Use VXdelegate : [" << params.Get<bool>("use_vxdelegate")
                   << "]";
}

TfLiteDelegatePtr VxDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  if (params.Get<bool>("use_vxdelegate")) {
    return evaluation::CreateVXDelegate();
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

}  // namespace tools
}  // namespace tflite
