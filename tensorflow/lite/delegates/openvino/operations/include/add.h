#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_ADD_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_ADD_H_

#include "tensorflow/lite/delegates/openvino/operations/operations_base.h"

namespace tflite {
namespace openvinodelegate {

class Add : public OperationsBase {
public:
    Add(int operationIndex) {}
    std::shared_ptr<ov::Node> createNode() override;
};

}  // namespace openvinodelegate
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_OPENVINO_ADD_H_
