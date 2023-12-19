#include "openvino_graph_builder.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus OpenVINOGraphBuilder::createNodeFromTfLiteOp(int node_id,
                                                          TfLiteRegistration* registration,
                                                          TfLiteNode* node,
                                                          TfLiteContext* context) {
    auto operationNode = createOpClass(node_id, registration);
    if (!operationNode) return kTfLiteError;
    operationNode->nodeManager = nodeManager;
    operationNode->SetContext(context);
    operationNode->UpdateNodeInfo(node->inputs->data, node->inputs->size, node->builtin_data);
    resultNode = operationNode->createNode();
    if (resultNode == nullptr) return kTfLiteError;
    nodeManager->setOutputAtOperandIndex(node->outputs->data[0], resultNode);
    return kTfLiteOk;
}
std::shared_ptr<OperationsBase> OpenVINOGraphBuilder::createOpClass(
    int operationIndex, TfLiteRegistration* registration) {
    switch (registration->builtin_code) {
        case kTfLiteBuiltinAdd: {
            return std::make_shared<Add>(operationIndex);
        }
        case kTfLiteBuiltinConv2d: {
            return std::make_shared<Conv2D>(operationIndex);
        }
        default:
            return nullptr;
    }
}

}  // namespace openvinodelegate
}  // namespace tflite
