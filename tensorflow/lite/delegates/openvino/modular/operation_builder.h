#include <openvino/openvino.hpp>

class OperationBuilder {
  virtual std::shared_ptr<ov::Node> void createNode() = 0;
  std::shared_ptr<ov::Node> getInputNode(int index);
  void addResultNode(int index, std::shared_ptr<ov::Node> result_node);
  std::shared_ptr<ov::Node> createConstNode(ov::element::Type elementType,
                                            ov::Shape shape, const void *data) {
    return std::make_shared<ov::opset8::Constant>(elementType, shape, data);


};
