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

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

#include <tensorflow/lite/kernels/ethos_u/ethosu.hpp>
#include <linux/ethosu.h>

using namespace std;
using namespace EthosU;


namespace tflite {
namespace ops {
namespace custom {
namespace ethosu {

//TfLite External Context shared with all the ethosu OP
struct TfLiteEthosuContext : public TfLiteExternalContext {
  shared_ptr<Buffer> arena_buffer;  //Output buffer for input/ouput/scratch tensor
  shared_ptr<Buffer> flash_buffer;  //Input buffer for weight tensor
  int num_references = 0;
};

TfLiteEthosuContext* GetEthosuContext(TfLiteContext* context) {
  return reinterpret_cast<TfLiteEthosuContext*>(
      context->GetExternalContext(context, kTfLiteEthosuContext));
}

TfLiteStatus Refresh(TfLiteContext* context) {
  return kTfLiteOk;
}

void IncrementUsageCounter(TfLiteContext* context) {
  auto* ptr = GetEthosuContext(context);
  if (ptr == nullptr) {
    ptr = new TfLiteEthosuContext;
    ptr->type = kTfLiteEthosuContext;
    ptr->Refresh = Refresh;
    ptr->num_references = 0;
    ptr->arena_buffer == nullptr;
    ptr->flash_buffer == nullptr;
    context->SetExternalContext(context, kTfLiteEthosuContext, ptr);
  }
  ptr->num_references++;
}

void DecrementUsageCounter(TfLiteContext* context) {
  auto* ptr = GetEthosuContext(context);
  if (ptr == nullptr) {
    TF_LITE_FATAL(
        "Call to DecrementUsageCounter() not preceded by "
        "IncrementUsageCounter()");
  }
  if (--ptr->num_references == 0) {
    ptr->arena_buffer == nullptr;
    ptr->flash_buffer == nullptr;
    delete ptr;
    context->SetExternalContext(context, kTfLiteEthosuContext, nullptr);
  }
}

#define BUFFER_ALIGNMENT 16
#define ALIGN_SIZE(size) ((size + BUFFER_ALIGNMENT - 1) & (~(BUFFER_ALIGNMENT - 1)))

struct OpData {
  Device* device;
  shared_ptr<Buffer> net_buffer;  //Buffer for cms tensor
  shared_ptr<Buffer> tensor_layout_buffer;   //Input buffer for data layout of in/out/scratch
  shared_ptr<Network> network;

  const int32_t* address_offsets;
  size_t cms_data_size;
  size_t flash_data_size;
  size_t arena_data_size;
  size_t output_data_size;
  bool enable_cycle_counter;
  vector<uint32_t> pmu_counter_config;
};

#define ETHOSU_DEFAULT_DEVICE_NAME (char*)"/dev/ethosu0"
#define OFFLINE_MEM_ALLOC_METADATA "OfflineMemoryAllocation"
#define DEFAULT_TIMEOUT 60000000000
#define CMS_TENSOR_INDEX 0
#define FLASH_TENSOR_INDEX 1
#define SCRATCH_TENSOR_INDEX 2
#define SCRATCH_FAST_TENSOR_INDEX 3
#define INPUT_TENSOR_INDEX 4

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* data = new OpData;

  char *device_name;
  device_name = getenv("ETHOSU_DEVICE_NAME");
  if (device_name == NULL)
      device_name = ETHOSU_DEFAULT_DEVICE_NAME;

  char *pmu_config;
  pmu_config = getenv("ETHOSU_PMU_CONFIG");
  if (pmu_config != NULL) {
       char *token = strtok(pmu_config, " ");
       int32_t event = (int32_t)atoi(token);
       if (event == 0) {
           TF_LITE_KERNEL_LOG(context, "Invalid Ethos-u PMU event! Ignore '%s'.\n", token);
       } else {
           data->pmu_counter_config.push_back(event);
       }

       while(token = strtok(NULL, " ")) {
          event = (int32_t)atoi(token);
          if (event == 0) {
              TF_LITE_KERNEL_LOG(context, "Invalid Ethos-u PMU event! Ignore '%s'.\n", token);
           } else if (data->pmu_counter_config.size() == ETHOSU_PMU_EVENT_MAX) {
              TF_LITE_KERNEL_LOG(context, "PMU out of bounds! Ignore '%s'.\n", token);
           } else {
              data->pmu_counter_config.push_back(event);
           }
       }
  }

  char *cycle_counter;
  cycle_counter = getenv("ETHOSU_ENABLE_CYCLE_COUNTER");
  if (cycle_counter != NULL) {
      if(strcmp(cycle_counter, "1") == 0) {
          data->enable_cycle_counter = true;
      } else if(strcmp(cycle_counter, "0") == 0) {
          data->enable_cycle_counter = false;
      } else {
          data->enable_cycle_counter = false;
          TF_LITE_KERNEL_LOG(context, "ETHOSU_ENABLE_CYCLE_COUNTER should be 0 or 1.\n");
      }
  } else {
      data->enable_cycle_counter = false;
  }

  try {
      data->device = Device::GetSingleton(device_name);
  } catch (std::exception &e) {
      delete data;
      data = nullptr;
  }
  IncrementUsageCounter(context);

  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  TFLITE_DCHECK(buffer != nullptr);
  OpData* data = reinterpret_cast<OpData*>(buffer);

  try {
      data->net_buffer = nullptr;
      data->tensor_layout_buffer = nullptr;
      data->device = nullptr;

      DecrementUsageCounter(context);
      delete data;
  } catch (std::exception &e) {
      TF_LITE_KERNEL_LOG(context, "Failed to release ethos_u buffers.\n");
  }
}


TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(context != nullptr);
  TF_LITE_ENSURE(context, node->inputs->size > 0);
  TFLITE_DCHECK(node->user_data != nullptr);
  TF_LITE_ENSURE(context, node->custom_initial_data_size > 0);
  auto ethosu_context = GetEthosuContext(context);

  OpData* data = static_cast<OpData*>(node->user_data);

  //Number of tenosr outputs + inputs + scrath + flash tensor
  uint32_t tensor_count = node->outputs->size + node->inputs->size - 1;
  //Get arena offset for each tensor from meta data
  const char* buffer = nullptr;
  size_t bytes;
  TF_LITE_ENSURE_OK(context, context->GetModelMetadata(context,
                    OFFLINE_MEM_ALLOC_METADATA, &buffer, &bytes));
  const uint32_t* metadata = reinterpret_cast<const uint32_t*>(buffer);
  if (bytes < 6 || metadata[2] < tensor_count + 1) {
      TF_LITE_KERNEL_LOG(context, "Failed to get address offsets from metadata\n");
      return kTfLiteError;
  }
  data->address_offsets = reinterpret_cast<const int32_t*>(&metadata[3]);

  size_t layout_buffer_size = 2 * sizeof(uint32_t) +            //Space for inputs/outputs tensor count
                              tensor_count * sizeof(uint32_t) + //Space for base_addr_size
                              tensor_count * sizeof(uint64_t);  //Space for the base_addr
  try {
      data->tensor_layout_buffer = make_shared<Buffer>(*data->device, layout_buffer_size);
  } catch (std::exception &e) {
      TF_LITE_KERNEL_LOG(context, "Failed to alloc ethos_u buffer.\n");
      return kTfLiteError;
  }
  uint32_t *tensor_layout_data = reinterpret_cast<uint32_t*>(data->tensor_layout_buffer->data());
  tensor_layout_data[0] = node->inputs->size - 4;
  tensor_layout_data[1] = node->outputs->size;
  uint32_t *base_addr_size = tensor_layout_data + 2;

  // Get command stream data size
  const TfLiteTensor* cms_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, CMS_TENSOR_INDEX, &cms_tensor));
  data->cms_data_size =cms_tensor->bytes;
  const TfLiteTensor* flash_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, FLASH_TENSOR_INDEX, &flash_tensor));
  data->flash_data_size = flash_tensor->bytes;
  base_addr_size[0] = static_cast<uint32_t>(data->flash_data_size);//flash tensor is first one

  data->arena_data_size = 0;
  // Get addresses to outputs data
  for (int i = 0; i < node->outputs->size; ++i) {
    TfLiteTensor* tensor;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i, &tensor));
    data->arena_data_size += ALIGN_SIZE(tensor->bytes);
    base_addr_size[i + node->inputs->size - 1] = tensor->bytes; //outputs tensor size is at last
    auto tensor_index = node->outputs->data[i];
  }
  data->output_data_size = data->arena_data_size;

  // Get addresses to inputs data
  for (int i = INPUT_TENSOR_INDEX; i < node->inputs->size; ++i) {
      const TfLiteTensor* tensor;
      TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, i, &tensor));
      base_addr_size[i - 1] = tensor->bytes; //inputs tensor
      data->arena_data_size += ALIGN_SIZE(tensor->bytes);
  }

  // Get addresses to scratch data
  for (int i = SCRATCH_TENSOR_INDEX; i < INPUT_TENSOR_INDEX; ++i) {
    const TfLiteTensor* tensor;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, i, &tensor));
    data->arena_data_size += ALIGN_SIZE(tensor->bytes);
    base_addr_size[i - 1] = tensor->bytes; //scratch tensor
  }


  try {
      data->net_buffer = make_shared<Buffer>(*data->device, data->cms_data_size);
      data->net_buffer->resize(data->cms_data_size);
      memcpy(data->net_buffer->data(), cms_tensor->data.raw, data->cms_data_size);

      if (data->flash_data_size != 0 && ethosu_context->flash_buffer == nullptr) {
          ethosu_context->flash_buffer = make_shared<Buffer>(*data->device, data->flash_data_size);
          memcpy(ethosu_context->flash_buffer->data(), flash_tensor->data.raw, data->flash_data_size);
      }

      if (ethosu_context->arena_buffer == nullptr
           || data->arena_data_size > ethosu_context->arena_buffer->capacity()) {
          ethosu_context->arena_buffer = make_shared<Buffer>(*data->device, data->arena_data_size);
      }
      data->network = make_shared<Network>(*data->device, data->net_buffer);
  } catch (std::exception &e) {
      TF_LITE_KERNEL_LOG(context, "Failed to alloc ethos_u buffer.\n");
      return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(context != nullptr);
  auto ethosu_context = GetEthosuContext(context);

  OpData* data = static_cast<OpData*>(node->user_data);
  char* arena_data = ethosu_context->arena_buffer->data();

  // Get addresses to input data, copy input data
  for (int i = INPUT_TENSOR_INDEX; i < node->inputs->size; ++i) {
    const TfLiteTensor* tensor;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, i, &tensor));

    int32_t addr_offset = data->address_offsets[node->inputs->data[i]];
    memcpy(arena_data + addr_offset, tensor->data.raw, tensor->bytes);
  }

  vector<shared_ptr<Buffer>> ifm {ethosu_context->arena_buffer, data->tensor_layout_buffer};
  vector<shared_ptr<Buffer>> ofm {};
  if (data->flash_data_size != 0) {
      ifm.push_back(ethosu_context->flash_buffer);
  }

  try {
      Inference inference(data->network, ifm.begin(), ifm.end(), ofm.begin(),
                     ofm.end(), data->pmu_counter_config, data->enable_cycle_counter);
      /* make sure the wait completes ok */
      if (inference.wait(DEFAULT_TIMEOUT) <= 0) {
          TF_LITE_KERNEL_LOG(context, "Ethos_u inference failed\n");
          return kTfLiteError;
      }
      /* Read out PMU counters if configured */
      if (data->pmu_counter_config.size() > 0) {
          const std::vector<uint32_t> pmus = inference.getPmuCounters();
          cout << "Ethos_u PMUs : [";
          for (auto p : pmus) {
              cout << " " << p;
          }
          cout << " ]" << endl;
      }
      if (data->enable_cycle_counter) {
          cout << "Ethos-u cycle counter: " << inference.getCycleCounter() << endl;
      }
  } catch (std::exception &e) {
      TF_LITE_KERNEL_LOG(context, "Failed to run ethos_u op inference.\n");
      return kTfLiteError;
  }


  // Get addresses to output data, copy output data
  for (int i = 0; i < node->outputs->size; ++i) {
    TfLiteTensor* tensor;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i, &tensor));

    int32_t addr_offset = data->address_offsets[node->outputs->data[i]];
    memcpy(tensor->data.raw, arena_data + addr_offset, tensor->bytes);
  }

  return kTfLiteOk;
}

}  // namespace ethosu

TfLiteRegistration* Register_ETHOSU() {
  static TfLiteRegistration r = {ethosu::Init,
                                 ethosu::Free,
                                 ethosu::Prepare,
                                 ethosu::Eval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}

const char* GetString_ETHOSU() { return "ethos-u"; }

}  // namespace custom
}  // namespace ops
}  // namespace tflite

