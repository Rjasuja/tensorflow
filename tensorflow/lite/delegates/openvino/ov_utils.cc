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
#include "tensorflow/lite/delegates/openvino/ov_utils.h"

namespace tflite {

bool CheckInputsType(int tensor_id, TfLiteContext *context,
                     TfLiteType expected_type) {
  TfLiteType type = context->tensors[tensor_id].type;
  return expected_type == type;
}
bool CheckNodeSupportByOpenVINO(const TfLiteRegistration *registration,
                                const TfLiteNode *node,
                                TfLiteContext *context) {
  switch (registration->builtin_code) {
  case kTfLiteBuiltinAdd: {
    int tensor_id1 = node->inputs->data[0];
    int tensor_id2 = node->inputs->data[1];
    if (CheckInputsType(tensor_id1, context, kTfLiteFloat32) &&
        CheckInputsType(tensor_id2, context, kTfLiteFloat32))
      return true;
    return false;
  }
  default:
    return false;
  }
}

} // namespace tflite
