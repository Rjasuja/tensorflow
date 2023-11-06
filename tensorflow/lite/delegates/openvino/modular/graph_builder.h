#include <openvino/openvino.hpp>
#include <openvino/opsets/opset3.hpp>

namespace tflite {
namespace openvinodelegate {

class GraphBuilder {
  void addInputParams(const int index);
  void setResultNode(const int index, std::shared_ptr<ov::Node> resultNode);
  void setOutputAtOperandIndex(int index, ov::Output<ov::Node> output);
  ov::Output<ov::Node> getOperationOutput(int index);
private:
  std::vector<ov::Output<ov::Node>> outputAtOperandIndex;
  std::vector<std::shared_ptr<ov::opset3::Parameter>> inputParams;
  std::vector<std::shared_ptr<ov::Node>> resultNodes;
  std::vector<int> compute_inputs;
  std::vector<int> outputs; 
};

}// namespace openvinodelegate
}// namespace tflite
