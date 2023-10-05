#include <vector>
#include <map>
#include <ie_cnn_network.h>
#include <openvino/runtime/core.hpp>

#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/delegates/openvino/NgraphNodes.h"

namespace tflite {
namespace openvinodelegate {
class OpenVINODelegateKernel : public SimpleDelegateKernelInterface {
  TfLiteStatus Init(TfLiteContext* context,
                            const TfLiteDelegateParams* params) override;

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override;

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override;

  TfLiteStatus CreateAddNode(TfLiteContext* context, int node_index,
                        TfLiteNode* node, const TfLiteTensor* tensors,
                        const TfLiteAddParams* add_params);

  TfLiteStatus CreateNode(TfLiteContext* context,
                        TfLiteRegistration* registration,
                        TfLiteNode* node, int node_index);

  std::shared_ptr<ov::Node> ApplyActivation(std::shared_ptr<ov::Node> input,
                        TfLiteFusedActivation activation);

  void addInputParams(const TfLiteContext* context, const int index);
 private:
  NgraphNodes* ngraphNodes;
  std::vector<std::shared_ptr<ov::opset3::Parameter>> inputParams = {};
  std::vector<std::shared_ptr<ov::Node>> resultNodes = {};
  ov::InferRequest inferRequest;
};
}
}