#include <memory>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/openvino/openvino_delegate.h"

namespace tflite {
std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>
AcquireOPENVINODelegate(int num_threads) {
  auto opts = TfLiteOpenVINODelegateOptionsDefault();
  // Note that we don't want to use the thread pool for num_threads == 1.
 return std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      TfLiteCreateOpenVINODelegate(&opts), TfLiteDeleteOpenVINODelegate);
}
}  // namespace tflite
