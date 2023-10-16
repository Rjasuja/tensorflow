
#include "openvino_delegate_kernel.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus OpenVINODelegateKernel::Init(TfLiteContext* context,
                            const TfLiteDelegateParams* params) {
    //TODO: REVISIT since this is needed by Eval(), create inputs and outputs object at global level for OpenVINODelegateKernel class
        TFLITE_LOG(ERROR) << "Openvino delegate Kernel Init function called" << "\n";
    const std::unordered_set<int> inputs(
      &params->input_tensors->data[0],
      &params->input_tensors->data[params->input_tensors->size]);
    std::unordered_set<int> outputs;
    for (int o = 0; o < params->output_tensors->size; o++) {
      const int output_tensor_idx = params->output_tensors->data[o];
      outputs.insert(output_tensor_idx);
    }

    TFLITE_LOG(INFO) << "Openvino delegate Kernel Ngraph object called" << "\n";
    ngraphNodes = new NgraphNodes();
    TFLITE_LOG(INFO) << "Openvino delegate Kernel Init function called" << "\n";

    for (auto i = inputs.begin(); i != inputs.end(); i++)
        addInputParams(context, *i);
    TFLITE_LOG(INFO) << "Openvino delegate Kernel Add input params function called" << "\n";

    std::vector<int> tensors(context->tensors_size, -1);
    for (int i = 0; i < params->nodes_to_replace->size; i++) {
      const int node_id = params->nodes_to_replace->data[i];
      TfLiteNode* delegate_node;
      TfLiteRegistration* delegate_node_registration;
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
            if (delegate_node_registration->builtin_code == kTfLiteBuiltinTransposeConv &&
                k == 0) {
              // Ignore the output size parameter (see above).
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

    for (int i = 0; i < params->nodes_to_replace->size; i++) {
      const int node_index = params->nodes_to_replace->data[i];

      TfLiteNode* node = nullptr;
      TfLiteRegistration* delegate_node_registration = nullptr;
      if (context->GetNodeAndRegistration(context, node_index, &delegate_node,
                                          &delegate_node_registration) != kTfLiteOk) {
        return kTfLiteError;
      }

      //TODO: Complete this implementation
      if (CreateNode(context, delegate_node_registration, delegate_node,
                    node_index) != kTfLiteOk) {
        return kTfLiteError;
      }

      for(int i = 0; i <= delegate_node->outputs->size; i++) {
        const int t = delegate_node->outputs->data[i];
        resultNodes.push_back(ngraphNodes->getOperationOutput(t).get_node_shared_ptr());
      }
    }

    ov::Core ie(std::string("/usr/local/lib64/plugins.xml"));
    std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(resultNodes, inputParams);
    ov::CompiledModel compiled_model;
    std::string deviceStr = "NPU";

    //TODO: get device string from flags
    if(model) {
      compiled_model = ie.compile_model(model, deviceStr);
//	    TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
//			"Network is loaded into device");

	    ov::pass::Manager manager;
	    manager.register_pass<ov::pass::Serialize>("/tmp/model.xml", "/tmp/model.bin");
	    manager.run_passes(model);
    }

    inferRequest = compiled_model.create_infer_request();
      //std::vector<int> node_inputs;
      //node_inputs.resize(delegate_node->inputs->size);
      //for (int j = 0; j < delegate_node->size; j++) {
      //  node_inputs.pushback(delegate_node->data[j]);
      //}
      //input_index_map.insert(std::pair(i, node_inputs));
    }
    return kTfLiteOk;
}

TfLiteStatus OpenVINODelegateKernel::Prepare(TfLiteContext* context, TfLiteNode* node) {

}

TfLiteStatus OpenVINODelegateKernel::Eval(TfLiteContext* context, TfLiteNode* node) {
    //TODO: complete this function
}

std::shared_ptr<ov::Node> OpenVINODelegateKernel::ApplyActivation(std::shared_ptr<ov::Node> input, TfLiteFusedActivation activation) {
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
//        TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
//            "unsupported fused activation (Sign) in node");
        return nullptr;
      case kTfLiteActSigmoid:
        return std::make_shared<ov::opset3::Sigmoid>(input);
      default:
//        TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
//                        "invalid fused activation (%d) in node",
//                        static_cast<int>(activation));
        return nullptr;
    }

}

void OpenVINODelegateKernel::addInputParams(const TfLiteContext* context, const int index) {
    const TfLiteTensor t = context->tensors[index];
    std::vector<size_t> dims(t.dims->data[0], t.dims->data[t.dims->size]);
    auto input = std::make_shared<ov::opset3::Parameter>(ov::element::f32, ov::Shape(dims.begin(), dims.end()));
    inputParams.push_back(input);
    ngraphNodes->setOutputAtOperandIndex(index, input);
}

TfLiteStatus OpenVINODelegateKernel::CreateAddNode(TfLiteContext* context, int node_index,
                      TfLiteNode* node, const TfLiteTensor* tensors,
                      const TfLiteAddParams* add_params) {
  const TfLiteTensor& input1_tensor = tensors[node->inputs->data[0]];
  const TfLiteTensor& input2_tensor = tensors[node->inputs->data[1]];
  auto inputNode1 = ngraphNodes->getInputNode(input1_tensor, node->inputs->data[0]);
  auto inputNode2 = ngraphNodes->getInputNode(input2_tensor, node->inputs->data[0]);
  auto addNode = std::make_shared<ov::opset8::Add>(inputNode1, inputNode2, ov::op::AutoBroadcastType::NUMPY);
  auto resultNode = ApplyActivation(addNode, add_params->activation);
  ngraphNodes->setOutputAtOperandIndex(node->outputs->data[0], resultNode);
}

TfLiteStatus OpenVINODelegateKernel::CreateNode(TfLiteContext* context,
                      TfLiteRegistration* registration,
                      TfLiteNode* node, int node_index) {
  switch (registration->builtin_code) {
    case kTfLiteBuiltinAdd: {
        const TfLiteAddParams* add_params =
            static_cast<const TfLiteAddParams*>(node->builtin_data);

        return CreateAddNode(context, node_index,
                            node, context->tensors, add_params);
      }
      default:
        return kTfLiteError;
    }
}
}
}
