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
#include "openvino_delegate_kernel.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus OpenVINODelegateKernel::Init(TfLiteContext *context,
                                          const TfLiteDelegateParams *params) {
  TFLITE_LOG(INFO) << "Openvino delegate Kernel Init function called"
                   << "\n";
  const std::unordered_set<int> inputs(
      &params->input_tensors->data[0],
      &params->input_tensors->data[params->input_tensors->size]);
  for (int o = 0; o < params->output_tensors->size; o++) {
    const int output_tensor_idx = params->output_tensors->data[o];
    outputs.insert(output_tensor_idx);
  }

  ngraphNodes = new NgraphNodes(inputs.size() + params->nodes_to_replace->size);

  std::vector<int> tensors(context->tensors_size, -1);
  for (int i = 0; i < params->nodes_to_replace->size; i++) {
    const int node_id = params->nodes_to_replace->data[i];
    TfLiteNode *delegate_node;
    TfLiteRegistration *delegate_node_registration;
    if (context->GetNodeAndRegistration(context, node_id, &delegate_node,
                                        &delegate_node_registration))
      return kTfLiteError;

    switch (delegate_node_registration->builtin_code) {
    case kTfLiteBuiltinMean:
    case kTfLiteBuiltinPad:
    case kTfLiteBuiltinReshape:
    case kTfLiteBuiltinResizeBilinear:
      // Ignore the second input (axes, static padding, or new shape),
      // because it is represented as parameters of the OpenVINO operator
      // rather than extra input.
      {
        const int t = delegate_node->inputs->data[0];
        tensors[t] = t;
      }
      break;
    case kTfLiteBuiltinSplit:
      // Ignore the first input (split_dim), as it is represented as
      // parameters of the OpenVINO operator rather than extra input.
      {
        const int t = delegate_node->inputs->data[1];
        tensors[t] = t;
        break;
      }
    default:
      // All other operators: process all inputs
      for (int k = 0; k < delegate_node->inputs->size; k++) {
        if (delegate_node_registration->builtin_code ==
                kTfLiteBuiltinTransposeConv &&
            k == 0) {
          continue;
        }
        const int t = delegate_node->inputs->data[k];
        if (t >= 0) {
          tensors[t] = t;
        }
      }
      for (int k = 0; k < delegate_node->outputs->size; k++) {
        const int t = delegate_node->outputs->data[k];
        if (t >= 0) {
          tensors[t] = t;
        }
      }
    }
  }

  // Filter out and remove -1 (unused) indexes.
  tensors.erase(std::remove_if(tensors.begin(), tensors.end(),
                               [](int i) { return i < 0; }),
                tensors.end());
  std::sort(tensors.begin(), tensors.end());

  for (int t : tensors) {
    const void *data = nullptr;
    if (context->tensors[t].allocation_type == kTfLiteMmapRo) {
      data = context->tensors[t].data.raw_const;
    }
    if (inputs.count(t) != 0) {
      if (data == nullptr) {
        compute_inputs.insert(t);
      }
    }
  }

  for (auto i = compute_inputs.begin(); i != compute_inputs.end(); i++) {
    if (context == nullptr)
      TFLITE_LOG(INFO) << "context is nullptr\n";
    addInputParams(context, *i);
  }

  for (int i = 0; i < params->nodes_to_replace->size; i++) {
    const int node_index = params->nodes_to_replace->data[i];

    TfLiteNode *delegate_node = nullptr;
    TfLiteRegistration *delegate_node_registration = nullptr;
    if (context->GetNodeAndRegistration(context, node_index, &delegate_node,
                                        &delegate_node_registration) !=
        kTfLiteOk) {
      return kTfLiteError;
    }

    if (CreateNode(context, delegate_node_registration, delegate_node,
                   node_index) != kTfLiteOk) {
      return kTfLiteError;
    }
  }

  resultNodes.resize(outputs.size());
  int it = 0;
  for (auto o : outputs) {
    resultNodes[it++] =
        ngraphNodes->getOperationOutput(o).get_node_shared_ptr();
  }

  ov::Core ie(std::string("/usr/local/lib64/plugins.xml"));
  std::shared_ptr<ov::Model> model =
      std::make_shared<ov::Model>(resultNodes, inputParams);
  ov::CompiledModel compiled_model;
  std::string deviceStr = "CPU";

  // TODO: get device string from flags
  if (model) {
    compiled_model = ie.compile_model(model, deviceStr);

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::Serialize>("/tmp/model.xml",
                                               "/tmp/model.bin");
    manager.run_passes(model);
  }

  inferRequest = compiled_model.create_infer_request();
  return kTfLiteOk;
}

TfLiteStatus OpenVINODelegateKernel::Prepare(TfLiteContext *context,
                                             TfLiteNode *node) {
  TFLITE_LOG(INFO) << "inside Prepare \n";
  return kTfLiteOk;
}

TfLiteStatus OpenVINODelegateKernel::Eval(TfLiteContext *context,
                                          TfLiteNode *node) {
  TFLITE_LOG(INFO) << "inside Eval \n";
  size_t i = 0;
  for (int t : compute_inputs) {
    ov::Tensor inputBlob = inferRequest.get_input_tensor(i++);
    uint8_t *dest = (uint8_t *)inputBlob.data<float>();
    auto len = context->tensors[t].bytes;
    void *srcPtr = context->tensors[t].data.data;
    float *src = (float *)srcPtr;
    std::memcpy((uint8_t *)dest, (uint8_t *)srcPtr, len);
  }
  inferRequest.start_async();
  inferRequest.wait_for(std::chrono::milliseconds(10000));
  size_t o = 0;
  for (int t : outputs) {
    ov::Tensor outputBlob = inferRequest.get_output_tensor(o);
    void *srcPtr = context->tensors[*(outputs.begin())].data.data;
    uint8_t *dest = (uint8_t *)outputBlob.data<float>();
    auto len = context->tensors[*(outputs.begin())].bytes;
    std::memcpy((void *)srcPtr, (void *)dest, len);
    o++;
  }

  return kTfLiteOk;
}

std::shared_ptr<ov::Node>
OpenVINODelegateKernel::ApplyActivation(std::shared_ptr<ov::Node> input,
                                        TfLiteFusedActivation activation) {
  switch (activation) {
  case kTfLiteActNone:
    return input;
  case kTfLiteActRelu:
    return std::make_shared<ov::opset3::Relu>(input);
  case kTfLiteActReluN1To1:
  case kTfLiteActRelu6:
    return std::make_shared<ov::opset3::Clamp>(input, 0, 6);
  case kTfLiteActTanh:
    return std::make_shared<ov::opset3::Tanh>(input);
  case kTfLiteActSignBit:
    return nullptr;
  case kTfLiteActSigmoid:
    return std::make_shared<ov::opset3::Sigmoid>(input);
  default:
    return nullptr;
  }
}

void OpenVINODelegateKernel::addInputParams(const TfLiteContext *context,
                                            const int index) {
  const TfLiteTensor t = context->tensors[index];
  std::vector<size_t> dims(t.dims->size);
  for (int i = 0; i < t.dims->size; i++) {
    dims[i] = t.dims->data[i];
  }
  auto input = std::make_shared<ov::opset3::Parameter>(
      ov::element::f32, ov::Shape(dims.begin(), dims.end()));
  if (input == NULL)
    TFLITE_LOG(INFO) << "addInputParams input node is null\n";
  ngraphNodes->setOutputAtOperandIndex(index, input);
  inputParams.push_back(input);
}

TfLiteStatus OpenVINODelegateKernel::CreateAddNode(
    TfLiteContext *context, int node_index, TfLiteNode *node,
    const TfLiteTensor *tensors, const TfLiteAddParams *add_params) {
  const TfLiteTensor &input1_tensor = tensors[node->inputs->data[0]];
  const TfLiteTensor &input2_tensor = tensors[node->inputs->data[1]];
  auto inputNode1 =
      ngraphNodes->getInputNode(input1_tensor, node->inputs->data[0]);
  if (inputNode1 == nullptr)
    TFLITE_LOG(INFO) << "input node 1 is null\n";
  auto inputNode2 =
      ngraphNodes->getInputNode(input2_tensor, node->inputs->data[1]);
  if (inputNode2 == nullptr)
    TFLITE_LOG(INFO) << "input Node 2 is null\n";
  auto addNode = std::make_shared<ov::opset8::Add>(
      inputNode1, inputNode2, ov::op::AutoBroadcastType::NUMPY);
  resultNode = ApplyActivation(addNode, add_params->activation);
  ngraphNodes->setOutputAtOperandIndex(node->outputs->data[0], resultNode);
  return kTfLiteOk;
}

TfLiteStatus
OpenVINODelegateKernel::CreateNode(TfLiteContext *context,
                                   TfLiteRegistration *registration,
                                   TfLiteNode *node, int node_index) {
  switch (registration->builtin_code) {
  case kTfLiteBuiltinAdd: {
    const TfLiteAddParams *add_params =
        static_cast<const TfLiteAddParams *>(node->builtin_data);
    return CreateAddNode(context, node_index, node, context->tensors,
                         add_params);
  }
  default:
    return kTfLiteError;
  }
}

} // namespace openvinodelegate
} // namespace tflite
