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
#ifndef TENSORFLOW_LITE_DELEGATES_NGRAPH_NODES_H_
#define TENSORFLOW_LITE_DELEGATES_NGRAPH_NODES_H_
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/tools/logging.h"
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset8.hpp>
#include <vector>
class NgraphNodes {
private:
  std::vector<ov::Output<ov::Node>> outputAtOperandIndex;

public:
  NgraphNodes(int operandsSize) {
    outputAtOperandIndex.resize(operandsSize + 1);
  }
  // REVISIT: Decide on the data type of index passed in these calls
  void setOutputAtOperandIndex(int index, ov::Output<ov::Node> output);
  ov::Output<ov::Node> getOperationOutput(int index);
  std::shared_ptr<ov::Node> getInputNode(const TfLiteTensor &tensor,
                                         int tensor_index);
  std::shared_ptr<ov::Node> createConstNode(ov::element::Type elementType,
                                            ov::Shape shape, const void *data) {
    return std::make_shared<ov::opset8::Constant>(elementType, shape, data);
  }
};

#endif
