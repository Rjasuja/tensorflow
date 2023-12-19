#include "tensorflow/lite/delegates/openvino/operations/include/add.h"

namespace tflite {
namespace openvinodelegate {

std::shared_ptr<ov::Node> Add::createNode() {
    TfLiteAddParams* add_params = (TfLiteAddParams*)GetBuiltinData();
    auto inputNode1 = getInputNode(tensor_indices[TFLITE_INPUT_NODE_1]);
    if (inputNode1 == nullptr) {
        TFLITE_LOG(INFO) << "input node 1 is null\n";
        return nullptr;
    }
    auto inputNode2 = getInputNode(tensor_indices[TFLITE_INPUT_NODE_2]);
    if (inputNode2 == nullptr) {
        TFLITE_LOG(INFO) << "input Node 2 is null\n";
        return nullptr;
    }
    auto addNode =
        std::make_shared<ov::opset8::Add>(inputNode1, inputNode2, ov::op::AutoBroadcastType::NUMPY);
    auto outputNode = ApplyActivation(addNode, add_params->activation);
    return outputNode;
}

}  // namespace openvinodelegate
}  // namespace tflite
