#ifndef TENSORFLOW_LITE_DELEGATES_OV_UTILS_H_
#define TENSORFLOW_LITE_DELEGATES_OV_UTILS_H_
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/builtin_ops.h"

namespace tflite {

//Return true if node is supported by OpenVINO runtime
bool CheckNodeSupportByOpenVINO(const TfLiteRegistration* registration,
                                         const TfLiteNode* node,
                                         TfLiteContext* context);

}
#endif  // TENSORFLOW_LITE_DELEGATES_OV_UTILS_H_
