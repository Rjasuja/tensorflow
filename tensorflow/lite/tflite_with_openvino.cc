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
#include <memory>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/openvino/openvino_delegate.h"

namespace tflite {
std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>
AcquireOPENVINODelegate(int num_threads) {
  auto opts = TfLiteOpenVINODelegateOptionsDefault();
  // Note that we don't want to use the thread pool for num_threads == 1.
  return std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      TfLiteOpenVINODelegateCreate(&opts), TfLiteOpenVINODelegateDelete);
}
}  // namespace tflite
