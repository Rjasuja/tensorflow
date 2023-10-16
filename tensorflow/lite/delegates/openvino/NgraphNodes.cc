#include "NgraphNodes.h"
void NgraphNodes::setOutputAtOperandIndex(int index, ov::Output<ov::Node> output) {
    outputAtOperandIndex[index] = output;
}

ov::Output<ov::Node> NgraphNodes::getOperationOutput(int index) {
    return outputAtOperandIndex[index];
}

ov::Output<ov::Node> NgraphNodes::getInputNode(const TfLiteTensor& tensor, int tensor_index) {
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
