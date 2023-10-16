#include "tensorflow/lite/delegates/openvino/ov_utils.h"

namespace tflite {

bool CheckInputsType(int tensor_id, TfLiteContext* context, TfLiteType expected_type) {
    TfLiteType type = context->tensors[tensor_id].type;
    return expected_type == type;
}
bool CheckNodeSupportByOpenVINO(const TfLiteRegistration* registration,
                                         const TfLiteNode* node,
                                         TfLiteContext* context) {
    switch (registration->builtin_code) {
        case kTfLiteBuiltinAdd: {
            int tensor_id1 = node->inputs->data[0];
            int tensor_id2 = node->inputs->data[1];
            if (CheckInputsType(tensor_id1, context, kTfLiteFloat32) &&
                CheckInputsType(tensor_id2, context, kTfLiteFloat32))
                return true;
            return false;
        }
        default:
            return false;
    }
}
}
