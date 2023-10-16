#ifndef TENSORFLOW_LITE_DELEGATES_NGRAPH_NODES_H_
#define TENSORFLOW_LITE_DELEGATES_NGRAPH_NODES_H_
#include <vector>
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset8.hpp>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/builtin_ops.h"
class NgraphNodes {
 private:
    std::vector<ov::Output<ov::Node>> outputAtOperandIndex = {};
 public:
    NgraphNodes() {}
    //REVISIT: Decide on the data type of index passed in these calls
    void setOutputAtOperandIndex(int index, ov::Output<ov::Node> output);
    ov::Output<ov::Node> getOperationOutput(int index);
    ov::Output<ov::Node> getInputNode(const TfLiteTensor& tensor, int tensor_index);
    std::shared_ptr<ov::Node> createConstNode(ov::element::Type elementType, ov::Shape shape,
		                              const void* data) {
	    return std::make_shared<ov::opset8::Constant>(elementType, shape, data);
    }
};

#endif
