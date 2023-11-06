#include "tensorflow/lite/c/common."
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace openvinodelegate {

class GraphManager {
  TfLiteStatus initializeGraph();
  TfLiteStatus generateGraph();
  void createInputParams();
private:
  GraphBuilder* graphBuilder;
  OperationBuilder* opBuilder;
  void instantitateOpClassFactory();
  std::vector<std::shared_ptr<OperationsBase>> opNodes;
  TfLiteContent* context;
  const TfLiteDelegateParams* params;
  std::vector<int> tensors_to_replace;
};

} //namespace openvinodelegate
} //namespace tflite
