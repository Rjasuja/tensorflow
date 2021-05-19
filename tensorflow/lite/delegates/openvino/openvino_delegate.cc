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

#include "tensorflow/lite/delegates/openvino/openvino_delegate.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/pass/serialize.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/opsets/opset3.hpp>
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/utils/sparsity_format_converter.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/tools/optimize/reduced_precision_support.h"

namespace tflite {
namespace openvino {
namespace {

// Forward declaration.
TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate);

class Delegate {
  friend class Subgraph;

 public:
  explicit Delegate(const TfLiteOpenVINODelegateOptions* options) {
    TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                         "Created TensorFlow Lite OpenVINO delegate for CPU.");

    options_ =
        options != nullptr ? *options : TfLiteOpenVINODelegateOptionsDefault();
  }

  TfLiteIntArray* PrepareOpsToDelegate(TfLiteContext* context);
  TfLiteDelegate* tflite_delegate() { return &delegate_; }

  bool support_signed_8bit_quantization() const {
    return (options_.flags & TFLITE_OPENVINO_DELEGATE_FLAG_QS8) != 0;
  }

  bool support_unsigned_8bit_quantization() const {
    return (options_.flags & TFLITE_OPENVINO_DELEGATE_FLAG_QU8) != 0;
  }

  bool support_any_8bit_quantization() const {
    return (options_.flags & (TFLITE_OPENVINO_DELEGATE_FLAG_QU8 |
                              TFLITE_OPENVINO_DELEGATE_FLAG_QS8)) != 0;
  }

  bool force_fp16() const {
    return (options_.flags & TFLITE_OPENVINO_DELEGATE_FLAG_FORCE_FP16) != 0;
  }

 private:
  TfLiteDelegate delegate_ = {
      reinterpret_cast<void*>(this),  // .data_
      DelegatePrepare,                // .Prepare
      nullptr,                        // .CopyFromBufferHandle
      nullptr,                        // .CopyToBufferHandle
      nullptr,                        // .FreeBufferHandle
      kTfLiteDelegateFlagsNone,       // .flags
  };

  TfLiteOpenVINODelegateOptions options_;
};

class NgraphNodes {
 private:
    std::vector<ov::Output<ov::Node>> outputAtOperandIndex = {};
 public:
    //REVISIT: Decide on the data type of index passed in these calls
    void setOutputAtOperandIndex(int index, ov::Output<ov::Node> output);
    ov::Output<ov::Node> getOperationOutput(int index);
    ov::Output<ov::Node> getInputNode(TfLiteTensor& tensor, int tensor_index);
    std::shared_ptr<ov::Node> createConstNode(ov::element::Type elementType, ov::Shape shape,
		                              const void* data) {
	    return std::make_shared<ov::opset8::Constant>(elementType, shape, data);
    }
};

void NgraphNodes::setOutputAtOperandIndex(int index, ov::Output<ov::Node> output) {
    outputAtOperandIndex[index] = output;
}

ov::Output<ov::Node> NgraphNodes::getOperationOutput(int index) {
    return outputAtOperandIndex[index];
}

ov::Output<ov::Node> NgraphNodes::getInputNode(TfLiteTensor& tensor, int tensor_index) {
    std::shared_ptr<ov::Node> input;
    if (tensor.type == kTfLiteFloat32) {
       ov::element::Type elementType;
       if(tensor.allocation_type == kTfLiteMmapRo) {
           const void* data = (const void*)tensor.data.raw_const;
	   //REVISIT : add support for other data types
           elementType = ov::element::f32;
	   ov::Shape tensor_shape;
           for (int i = 0; i < tensor.dims->size; i++)
               tensor_shape.push_back(static_cast<size_t>(tensor.dims->data[i]));
	   input = createConstNode(elementType, tensor_shape, data);
       }
    } else {
        input = getOperationOutput(tensor_index).get_node_shared_ptr();
    }

    return input;
}

void addInputParams(const TfLiteContext* context, const int index) {
    const TfLiteTensor t = context->tensors[index];
    std::vector<size_t> dims(t.dims->data[0], t.dims->data[NumDimensions(&t)]);
    auto input = std::make_shared<ov::opset3::Parameter>(ov::element::f32, ov::Shape(dims.begin(), dims.end()));
    inputParams.insert(input);
    ngraphNodes->setOutputAtOperandIndex(index, input);
}

class Subgraph {
 public:
  static Subgraph* Create(TfLiteContext* context,
                          const TfLiteDelegateParams* params,
                          const Delegate& delegate) {
    // Convert subgraph inputs and outputs to hash sets for faster lookup.
    const std::unordered_set<int> inputs(
        &params->input_tensors->data[0],
        &params->input_tensors->data[params->input_tensors->size]);
    std::unordered_set<int> outputs;
    for (int o = 0; o < params->output_tensors->size; o++) {
      const int output_tensor_idx = params->output_tensors->data[o];
      outputs.insert(output_tensor_idx);
    }
    for (auto i = inputs.begin(); i != inputs.end(); i++)
        addInputParams(context, i);

    TfLiteIntArray* execution_plan;
    if (context->GetExecutionPlan(context, &execution_plan) != kTfLiteOk) {
      return nullptr;
    }


    // Detect which tensors are used as inputs or outputs of any subgraph nodes.
    // -1 denotes tensor not used in the subgraph. These indexes will be
    // filtered out and removed later.
    std::vector<int> tensors(context->tensors_size, -1);
    for (int i = 0; i < params->nodes_to_replace->size; i++) {
      const int node_index = params->nodes_to_replace->data[i];

      TfLiteNode* node = nullptr;
      TfLiteRegistration* registration = nullptr;
      if (context->GetNodeAndRegistration(context, node_index, &node,
                                          &registration) != kTfLiteOk) {
        return nullptr;
      }

      switch (registration->builtin_code) {
        case kTfLiteBuiltinMean:
        case kTfLiteBuiltinPad:
        case kTfLiteBuiltinReshape:
        case kTfLiteBuiltinResizeBilinear:
          // Ignore the second input (axes, static padding, or new shape),
          // because it is represented as parameters of the OpenVINO operator
          // rather than extra input.
          {
            const int t = node->inputs->data[0];
            tensors[t] = t;
          }
          break;
        case kTfLiteBuiltinSplit:
          // Ignore the first input (split_dim), as it is represented as
          // parameters of the OpenVINO operator rather than extra input.
          {
            const int t = node->inputs->data[1];
            tensors[t] = t;
            break;
          }
        default:
          // All other operators: process all inputs
          for (int k = 0; k < node->inputs->size; k++) {
            if (registration->builtin_code == kTfLiteBuiltinTransposeConv &&
                k == 0) {
              // Ignore the output size parameter (see above).
              continue;
            }
            const int t = node->inputs->data[k];
            if (t >= 0) {
              tensors[t] = t;
            }
          }
      }
      for (int k = 0; k < node->outputs->size; k++) {
        const int t = node->outputs->data[k];
        if (t >= 0) {
          tensors[t] = t;
        }
      }
    }
    // Filter out and remove -1 (unused) indexes.
    tensors.erase(std::remove_if(tensors.begin(), tensors.end(),
                                 [](int i) { return i < 0; }),
                  tensors.end());
    std::sort(tensors.begin(), tensors.end());

    std::unordered_set<int> compute_inputs;
    for(int t : tensors) {
	const void* data = nullptr;
        if(context->tensors[t].allocation_type != kTfLiteMmapRo) {
            data = (const void*)context->tensors[t].data.raw_const;
        }
	if(inputs.count(t) != 0) {
            if(data == nullptr)
                compute_inputs.insert(t);
	}
    }

    // Create ngraph nodes for TFLite delegate nodes
    for (int i = 0; i < params->nodes_to_replace->size; i++) {
      const int node_index = params->nodes_to_replace->data[i];

      TfLiteNode* node = nullptr;
      TfLiteRegistration* registration = nullptr;
      if (context->GetNodeAndRegistration(context, node_index, &node,
                                          &registration) != kTfLiteOk) {
        return nullptr;
      }

      if (VisitNode(delegate, context, registration, node,
                    node_index, false) != kTfLiteOk) {
        return nullptr;
      }

      for(int i = 0; i <= node->outputs->size; i++) {
        const int t = node->outputs->data[i];
        resultNodes.insert(getOperationOutput(t));
      }
    }

    //TODO REVISIT: Set Result Nodes 
    ov::Core ie(std::string("/usr/local/lib64/plugins.xml"));
    std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(resultNodes, inputParams);
    ov::CompiledModel compiled_model;
    std::string deviceStr = "NPU";

    //TODO: get device string from flags
    if(model) {
        compiled_model = ie.compile_model(model, deviceStr);
	TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
			"Network is loaded into device");

	ov::pass::Manager manager;
	manager.register_pass<ov::pass::Serialize>("/tmp/model.xml", "/tmp/model.bin");
	manager.run_passes(model);
    }

    inferRequest = compiled_model.create_infer_request();

    return new Subgraph(delegate, model, compute_inputs, outputs);
  }

  TfLiteStatus Prepare(TfLiteContext* context) { return kTfLiteOk; }

  TfLiteStatus Invoke(TfLiteContext* context) {
    size_t i = 0;
    for (int t : graph_inputs_) {
        ov::Tensor inputBlob = inferRequest.get_input_tensor(i++);
	uint8_t* dest = (uint8_t*)inputBlob.data<float>();
	auto len = context->tensors[t].bytes;
	void* srcPtr = context->tensors[t].data.data;
	std::memcpy((uint8_t*)dest, (uint8_t*)srcPtr, len);
    }
    inferRequest.start_async();
    size_t o = 0;
    for (int t : graph_outputs_) {
        ov::Tensor outputBlob = inferRequest.get_output_tensor(o++);
	void* srcPtr = context->tensors[t].data.data;
	uint8_t* dest = (uint8_t*)outputBlob.data<float>();
	auto len = context->tensors[t].bytes;
	std::memcpy((void*)srcPtr, (void*) dest, len);
    }

    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorNonDynamicAllocation(
      TfLiteContext* context, const TfLiteTensor& tensor, int tensor_index,
      int node_index) {
    // TODO: remove checks once dynamic tensors are supported
    if (tensor.allocation_type == kTfLiteDynamic) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context,
          "invalid allocation type in tensor #%d in node #%d: "
          "expected non-dynamic tensor",
          tensor_index, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckActivation(int node_index,
                                      TfLiteFusedActivation activation) {
    switch (activation) {
      case kTfLiteActNone:
        return kTfLiteOk;
      case kTfLiteActRelu:
        return kTfLiteOk;
      case kTfLiteActReluN1To1:
      case kTfLiteActRelu6:
        return kTfLiteOk;
      case kTfLiteActTanh:
        return kTfLiteOk;
      case kTfLiteActSignBit:
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
            "unsupported fused activation (Sign) in node #%d",
            node_index);
        return kTfLiteError;
      case kTfLiteActSigmoid:
        return kTfLiteOk;
      default:
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                        "invalid fused activation (%d) in node #%d",
                        static_cast<int>(activation), node_index);
        return kTfLiteError;
    }
  }
 
  static TfLiteStatus CheckNumInputsAndOutputs(TfLiteContext* context,
                                               TfLiteNode* node,
                                               int expected_num_inputs,
                                               int expected_num_outputs,
                                               int node_index) {
    if (node->inputs->size != expected_num_inputs) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "unexpected number of inputs (%d != %d) in node #%d",
          node->inputs->size, expected_num_inputs, node_index);
      return kTfLiteError;
    }
    if (node->outputs->size != expected_num_outputs) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "unexpected number of outputs (%d != %d) in node #%d",
          node->outputs->size, expected_num_outputs, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorFloat32OrQUInt8Type(const Delegate& delegate,
                                                     TfLiteContext* context,
                                                     const TfLiteTensor& tensor,
                                                     int tensor_index,
                                                     int node_index) {
    switch (tensor.type) {
      case kTfLiteFloat32:
        return kTfLiteOk;
      case kTfLiteInt8:
        if (delegate.support_signed_8bit_quantization()) {
          const auto* quantization_params =
              static_cast<const TfLiteAffineQuantization*>(
                  tensor.quantization.params);
          if (tensor.quantization.type != kTfLiteAffineQuantization ||
              quantization_params->quantized_dimension != 0 ||
              quantization_params->scale == nullptr ||
              quantization_params->scale->size != 1) {
            TF_LITE_MAYBE_KERNEL_LOG(
                context,
                "unsupported quantization type %d in tensor #%d in node #%d",
                tensor.quantization.type, tensor_index, node_index);
            return kTfLiteError;
          }
          return kTfLiteOk;
        }
        break;
      case kTfLiteUInt8:
        if (delegate.support_unsigned_8bit_quantization()) {
          const auto* quantization_params =
              static_cast<const TfLiteAffineQuantization*>(
                  tensor.quantization.params);
          if (tensor.quantization.type != kTfLiteAffineQuantization ||
              quantization_params->quantized_dimension != 0 ||
              quantization_params->scale == nullptr ||
              quantization_params->zero_point == nullptr ||
              quantization_params->scale->size != 1 ||
              quantization_params->zero_point->size != 1) {
            TF_LITE_MAYBE_KERNEL_LOG(
                context,
                "unsupported quantization type %d in tensor #%d in node #%d",
                tensor.quantization.type, tensor_index, node_index);
            return kTfLiteError;
          }
          return kTfLiteOk;
        }
        break;
      default:
        break;
    }
    TF_LITE_MAYBE_KERNEL_LOG(
        context, "unsupported type %s in tensor #%d in node #%d",
        TfLiteTypeGetName(tensor.type), tensor_index, node_index);
    return kTfLiteError;
  }


  static TfLiteStatus VisitNode(
      const Delegate& delegate, TfLiteContext* context,
      TfLiteRegistration* registration, TfLiteNode* node, int node_index,
      const bool detect_supported_op) {
    // TFLite context used for logging purposes. When we create a new node
    // (subgraph is non-null), logging context is the same as context, and error
    // messages are passed to TFLite. When we detect supported operations
    // (subgraph is null), logging context is null, and error messages are
    // supressed.
    TfLiteContext* logging_context = detect_supported_op ? nullptr : context;
    switch (registration->builtin_code) {
     case kTfLiteBuiltinAdd: {
        const TfLiteAddParams* add_params =
            static_cast<const TfLiteAddParams*>(node->builtin_data);

        return VisitAddNode(delegate, logging_context, node_index,
                            node, context->tensors, add_params,
                            detect_supported_op);
      }
      default:
        return kTfLiteError;
    }
  }

  std::shared_ptr<ov::Node> ApplyActivation(std::shared_ptr<ov::Node> input, TfLiteFusedActivation activation) {
    switch (activation) {
      case kTfLiteActNone:
        return input;
      case kTfLiteActRelu:
        return std::make_shared<ov::opset3::Relu>(input);
      case kTfLiteActReluN1To1:
      case kTfLiteActRelu6:
        return std::make_shared<ov::opset3::Clamp>(input,0, 6);
      case kTfLiteActTanh:
        return std::make_shared<ov::opset3::Tanh>(input);
      case kTfLiteActSignBit:
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
            "unsupported fused activation (Sign) in node");
        return nullptr;
      case kTfLiteActSigmoid:
        return std::make_shared<ov::opset3::Sigmoid>(input);
      default:
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                        "invalid fused activation (%d) in node",
                        static_cast<int>(activation));
        return nullptr;
    }

  }

  static TfLiteStatus VisitAddNode(
      const Delegate& delegate,
      TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteAddParams* add_params,
      const bool detect_supported_op) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 2, 1, node_index));

    const TfLiteTensor& input1_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, input1_tensor,
                                       node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input1_tensor, node->inputs->data[0], node_index));

    const TfLiteTensor& input2_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, input2_tensor,
                                       node->inputs->data[1], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input2_tensor, node->inputs->data[1], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, output_tensor,
                                       node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, node->outputs->data[0], node_index));

    if (detect_supported_op) {
      if (add_params != nullptr) {
        TF_LITE_ENSURE_STATUS(
            CheckActivation(node_index, add_params->activation));
      }
    } else {
      auto inputNode1 = ngraphNodes->getInputNode(input1_tensor, node->inputs->data[0]);
      auto inputNode2 = ngraphNodes->getInputNode(input2_tensor, node->inputs->data[0]);
      auto addNode = std::make_shared<ov::opset8::Add>(inputNode1, inputNode2, ov::op::AutoBroadcastType::NUMPY);
      auto resultNode = ApplyActivation(addNode, add_params->activation);
      ngraphNodes->setOutputAtOperandIndex(node->outputs->data[0], resultNode);
    }

    return kTfLiteOk;
  }

 private:
  Subgraph(const Delegate& delegate, const std::shared_ptr<const ov::Model>& model,
           const std::unordered_set<int>& graph_inputs,
           const std::unordered_set<int>& graph_outputs)
      : model_(model),
        graph_inputs_(graph_inputs),
        graph_outputs_(graph_outputs) {
    for (int t : graph_outputs) {
        t = nullptr;
    }
  }

  std::shared_ptr<NgraphNodes> ngraphNodes;
  const std::shared_ptr<const ov::Model> model_;
  std::unordered_set<int> graph_outputs_;
  std::unordered_set<int> graph_inputs_;
  //TODO: Remove if not needed
  // Memory location to use for 0-size extenal tensors, as TFLite init their
  // data pointer to nullptr, and OpenVINO requires valid data pointers.
  char dummy_data_{0};
  void addInputParams(const TfLiteContext* context, const int index);
  void addResultNode(const TfLiteContext* context, const int index);
  std::vector<std::shared_ptr<ov::opset3::Parameter>> inputParams;
  std::vector<std::shared_ptr<ov::Node>> resultNodes;
  ov::InferRequest inferRequest;
};

TfLiteIntArray* Delegate::PrepareOpsToDelegate(TfLiteContext* context) {
  //TODO: Clear previous data, in case the delegate is reused without re-creation.

  TfLiteIntArray* execution_plan = nullptr;
  if (context->GetExecutionPlan(context, &execution_plan) != kTfLiteOk) {
    TF_LITE_KERNEL_LOG(context, "Unable to get graph execution plan.");
    return nullptr;
  }

  TfLiteIntArray* nodes_to_delegate =
      TfLiteIntArrayCreate(execution_plan->size);
  nodes_to_delegate->size = 0;
  for (int i = 0; i < execution_plan->size; ++i) {
    const int node_index = execution_plan->data[i];

    // Check if TFLite nodes can be delegated to OpenVINO
    TfLiteNode* node = nullptr;
    TfLiteRegistration* registration = nullptr;
    if (context->GetNodeAndRegistration(context, node_index, &node,
                                        &registration) != kTfLiteOk) {
      TF_LITE_KERNEL_LOG(context,
                         "Unable to get node and registration for node %d.",
                         node_index);
      continue;  // Soft error (skip this node).
    }


    if (Subgraph::VisitNode(/*delegate=*/*this, context,
                            registration, node, node_index, true) != kTfLiteOk) {
      continue;
    }

    nodes_to_delegate->data[nodes_to_delegate->size++] = node_index;
  }

  std::sort(&nodes_to_delegate->data[0],
            &nodes_to_delegate->data[nodes_to_delegate->size]);

#ifdef OPENVINO_DELEGATE_TEST_MODE
  // In the test mode build (used by unit tests), OPENVINO delegate claims to
  // support all operators in the execution plan to disable fallback to the
  // default TensorFlow Lite kernels. Thus, if any of the ops in the model are
  // not supported by the delegate, they will cause a failure in
  // ::tflite::Interpreter::ModifyGraphWithDelegate, to be caught in the unit
  // tests.
  nodes_to_delegate->size = execution_plan->size;
  std::copy(&execution_plan->data[0],
            &execution_plan->data[execution_plan->size],
            &nodes_to_delegate->data[0]);
#endif

  return nodes_to_delegate;
}

void* SubgraphInit(TfLiteContext* context, const char* buffer, size_t length) {
  const TfLiteDelegateParams* params =
      reinterpret_cast<const TfLiteDelegateParams*>(buffer);

  return static_cast<void*>(Subgraph::Create(
      context, params,
      *static_cast<::tflite::openvino::Delegate*>(params->delegate->data_)));
}

TfLiteStatus SubgraphPrepare(TfLiteContext* context, TfLiteNode* node) {
  if (node->user_data == nullptr) {
    return kTfLiteError;
  }

  return static_cast<Subgraph*>(node->user_data)->Prepare(context);
}

TfLiteStatus SubgraphInvoke(TfLiteContext* context, TfLiteNode* node) {
  if (node->user_data == nullptr) {
    return kTfLiteError;
  }

  return static_cast<Subgraph*>(node->user_data)->Invoke(context);
}

void SubgraphFree(TfLiteContext* context, void* buffer) {
  if (buffer != nullptr) {
    delete static_cast<Subgraph*>(buffer);
  }
}

const TfLiteRegistration kSubgraphRegistration = {
    /*.init=*/SubgraphInit,
    /*.free=*/SubgraphFree,
    /*.prepare=*/SubgraphPrepare,
    /*.invoke=*/SubgraphInvoke,
    /*.profiling_string=*/nullptr,
    /*.builtin_code=*/0,
    /*.custom_name=*/"TfLiteOpenVINODelegate",
    /*.version=*/2,
};

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  TfLiteIntArray* ops_to_replace =
      static_cast<::tflite::openvino::Delegate*>(delegate->data_)
          ->PrepareOpsToDelegate(context);
  if (ops_to_replace == nullptr) {
    return kTfLiteError;
  }

  const TfLiteStatus status = context->ReplaceNodeSubsetsWithDelegateKernels(
      context, kSubgraphRegistration, ops_to_replace, delegate);
  TfLiteIntArrayFree(ops_to_replace);
  return status;
}

}  // namespace
}  // namespace openvino
}  // namespace tflite


TfLiteOpenVINODelegateOptions TfLiteOpenVINODelegateOptionsDefault() {
  TfLiteOpenVINODelegateOptions options = {0};

  // Quantized inference is enabled by default on Web platform
#ifdef OPENVINO_DELEGATE_ENABLE_QS8
  options.flags |= TFLITE_OPENVINO_DELEGATE_FLAG_QS8;
#endif
#ifdef OPENVINO_DELEGATE_ENABLE_QU8
  options.flags |= TFLITE_OPENVINO_DELEGATE_FLAG_QU8;
#endif

  // Enable quantized inference for the delegate build used in unit tests.
#ifdef OPENVINO_DELEGATE_TEST_MODE
  options.flags |= TFLITE_OPENVINO_DELEGATE_FLAG_QS8;
  options.flags |= TFLITE_OPENVINO_DELEGATE_FLAG_QU8;
#endif  // OPENVINO_DELEGATE_TEST_MODE

  return options;
}

TfLiteDelegate* TfLiteOpenVINODelegateCreate(
    const TfLiteOpenVINODelegateOptions* options) {
  //TODO:: Do openvino initialization specific steps here if any

  auto* openvino_delegate = new ::tflite::openvino::Delegate(options);
  return openvino_delegate ? openvino_delegate->tflite_delegate() : nullptr;
}

void TfLiteOpenVINODelegateDelete(TfLiteDelegate* delegate) {
  if (delegate != nullptr) {
    delete static_cast<::tflite::openvino::Delegate*>(delegate->data_);
  }
}
