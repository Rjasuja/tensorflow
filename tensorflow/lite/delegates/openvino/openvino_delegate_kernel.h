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
#include <map>
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/serialize.hpp>
#include <openvino/runtime/core.hpp>
#include <vector>

#include "openvino_delegate_core.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace openvinodelegate {
class OpenVINODelegateKernel : public SimpleDelegateKernelInterface {
public:
    explicit OpenVINODelegateKernel()
        : ov_delegate_manager(std::make_unique<OpenVINODelegateManager>("")) {}
    TfLiteStatus Init(TfLiteContext* context, const TfLiteDelegateParams* params) override;

    TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override;

    TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override;

    std::shared_ptr<ov::Node> ApplyActivation(std::shared_ptr<ov::Node> input,
                                              TfLiteFusedActivation activation);

private:
    std::unique_ptr<OpenVINODelegateManager> ov_delegate_manager;
};

}  // namespace openvinodelegate
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_OPENVINO_DELEGATE_KERNEL_H_
