// openvino_graph_builder.cc
#include "openvino_delegate_core.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus OpenVINODelegateManager::createGraphfromTfLite(TfLiteContext* context,
                                                            const TfLiteDelegateParams* params) {
    const std::unordered_set<int> inputs(&params->input_tensors->data[0],
                                         &params->input_tensors->data[params->input_tensors->size]);
    openvino_graph_builder = std::make_unique<OpenVINOGraphBuilder>();

    for (int o = 0; o < params->output_tensors->size; o++) {
        const int output_tensor_idx = params->output_tensors->data[o];
        outputs.push_back(output_tensor_idx);
    }

    for (int i = 0; i < params->nodes_to_replace->size; i++) {
        const int delegate_node_id = params->nodes_to_replace->data[i];
        TfLiteNode* delegate_node;
        TfLiteRegistration* delegate_node_registration;
        if (context->GetNodeAndRegistration(context, delegate_node_id, &delegate_node,
                                            &delegate_node_registration))
            return kTfLiteError;

        for (int k = 0; k < delegate_node->inputs->size; k++) {
            if (delegate_node_registration->builtin_code == kTfLiteBuiltinTransposeConv && k == 0) {
                continue;
            }
            const int t = delegate_node->inputs->data[k];
            const void* data = nullptr;
            if (context->tensors[t].allocation_type == kTfLiteMmapRo) {
                data = context->tensors[t].data.raw_const;
                if (openvino_graph_builder->createConstNode(context, t) != kTfLiteOk)
                    return kTfLiteError;
            }
            if (inputs.count(t) != 0) {
                if (data == nullptr) {
                    if (openvino_graph_builder->addInputParams(context, t) != kTfLiteOk)
                        return kTfLiteError;
                    compute_inputs.push_back(t);
                }
            }
        }
        if (openvino_graph_builder->createNodeFromTfLiteOp(
                delegate_node_id, delegate_node_registration, delegate_node, context) != kTfLiteOk)
            return kTfLiteError;
    }

    openvino_graph_builder->updateResultNodes(outputs);
    std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(
        openvino_graph_builder->getResultNodes(), openvino_graph_builder->getInputParams());
    // TODO: get device string from flags
    std::string deviceStr = "CPU";
    if (model) {
        compiled_model = openvino_delegate_core.compile_model(model, deviceStr);

        ov::pass::Manager manager;
        manager.register_pass<ov::pass::Serialize>("/tmp/model.xml", "/tmp/model.bin");
        manager.run_passes(model);
    }

    inferRequest = compiled_model.create_infer_request();
    return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
