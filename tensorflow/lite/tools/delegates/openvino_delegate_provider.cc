#include <string>
#include <utility>

#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

namespace tflite {
namespace tools {

class OpenVINODelegateProvider : public DelegateProvider {
 public:
  OpenVINODelegateProvider() {
    default_params_.AddParam("use_openvino", ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final { return "OPENVINO"; }
};
REGISTER_DELEGATE_PROVIDER(OpenVINODelegateProvider);

std::vector<Flag> OpenVINODelegateProvider::CreateFlags(
    ToolParams* params) const {
  std::vector<Flag> flags = {CreateFlag<bool>(
      "use_openvino", params,
      "explicitly apply the OPENVINO delegate. Note the OPENVINO delegate could "
      "be implicitly applied by the TF Lite runtime regardless the value of "
      "this parameter. To disable this implicit application, set the value to "
      "false explicitly.")};
  return flags;
}

void OpenVINODelegateProvider::LogParams(const ToolParams& params,
                                        bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_openvino", "Use openvino", verbose);
}


//TODO
TfLiteDelegatePtr OpenVINODelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  if (params.Get<bool>("use_openvino"))
    return ::tflite::evaluation::CreateOPENVINODelegate();
  else
    return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

std::pair<TfLiteDelegatePtr, int>
OpenVINODelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr),
                        params.GetPosition<bool>("use_openvino"));
}
}  // namespace tflite

}  // namespace tools
