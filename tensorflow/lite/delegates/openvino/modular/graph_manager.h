#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/tools/logging.h"
#include "graph_builder.h"
#include "operation_builder.h"

namespace tflite {
namespace openvinodelegate {

class GraphManager {
  TfLiteStatus initializeGraph();
  TfLiteStatus generateGraph();
  void createInputParams();
  void GraphManager(TfLiteContext* context, const TfLiteDelegateParams* params) : params_(params), context_(context) {}
private:
  GraphBuilder* graphBuilder;
  OperationBuilder* opBuilder;
  //to instantiate classes for each op before actually calling createNode() for that op
  void instantiateOpClassFactory();
  std::vector<std::shared_ptr<OperationBuilder>> opNodes;
  TfLiteContent* context_;
  const TfLiteDelegateParams* params_;
  std::vector<int> tensors_to_replace;
};

} //namespace openvinodelegate
} //namespace tflite
