#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_NODE_MANAGER_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_NODE_MANAGER_H_

#include <openvino/openvino.hpp>

class NodeManager {
public:
    NodeManager() {}
    std::shared_ptr<ov::Node> getInterimNodeOutput(int index) {
        auto node = outputAtOperandIndex[index];
        return node.get_node_shared_ptr();
    }
    void setOutputAtOperandIndex(int index, ov::Output<ov::Node> output) {
        outputAtOperandIndex.insert(std::pair<int, ov::Output<ov::Node>>(index, output));
    }

private:
    std::map<int, ov::Output<ov::Node>> outputAtOperandIndex;
};
#endif  // TENSORFLOW_LITE_DELEGATES_OPENVINO_NODE_MANAGER_H_
