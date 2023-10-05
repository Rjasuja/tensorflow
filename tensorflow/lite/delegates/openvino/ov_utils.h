#include "tensorflow/lite/c/common.h"

namespace tflite {

//Return true if node is supported by OpenVINO runtime
bool CheckNodeSupportByOpenVINO(const TfLiteRegistration* registration,
                                         const TfLiteNode* node,
                                         TfLiteContext* context);

}