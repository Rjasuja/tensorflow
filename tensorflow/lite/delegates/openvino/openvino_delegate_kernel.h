/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_DELEGATE_KERNEL_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_DELEGATE_KERNEL_H_
#include "tensorflow/lite/c/common.h"
#include <ie_cnn_network.h>
#include <map>
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/serialize.hpp>
#include <openvino/runtime/core.hpp>
#include <vector>

#include "NgraphNodes.h"
#include "ov_utils.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace openvinodelegate {
class OpenVINODelegateKernel : public SimpleDelegateKernelInterface {
public:
  explicit OpenVINODelegateKernel() {}
  TfLiteStatus Init(TfLiteContext *context,
                    const TfLiteDelegateParams *params) override;

  TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) override;

  TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) override;

  TfLiteStatus CreateAddNode(TfLiteContext *context, int node_index,
                             TfLiteNode *node, const TfLiteTensor *tensors,
                             const TfLiteAddParams *add_params);

  TfLiteStatus CreateNode(TfLiteContext *context,
                          TfLiteRegistration *registration, TfLiteNode *node,
                          int node_index);

  std::shared_ptr<ov::Node> ApplyActivation(std::shared_ptr<ov::Node> input,
                                            TfLiteFusedActivation activation);

  void addInputParams(const TfLiteContext *context, const int index);

private:
  NgraphNodes *ngraphNodes;
  std::vector<std::shared_ptr<ov::opset3::Parameter>> inputParams = {};
  std::vector<std::shared_ptr<ov::Node>> resultNodes = {};
  std::shared_ptr<ov::Node> resultNode;
  ov::InferRequest inferRequest;
  std::unordered_set<int> compute_inputs;
  std::unordered_set<int> outputs;
};

} // namespace openvinodelegate
} // namespace tflite
#endif // TENSORFLOW_LITE_DELEGATES_OPENVINO_DELEGATE_KERNEL_H_
