#include <ie_cnn_network.h>
#include "openvino/runtime/core.hpp"

#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include "tensorflow/lite/builtin_ops.h"
#include "openvino_delegate.h"

namespace tflite {
namespace openvinodelegate {
class OpenVINODelegate : public SimpleDelegateInterface {
    public:
    explicit OpenVINODelegate(const TfLiteOpenVINODelegateOptions* options) {
	options_ = *options;
        TFLITE_LOG(INFO) << "Openvino delegate object created" << "\n";
        if (options == nullptr)
            options_ = TfLiteOpenVINODelegateOptionsDefault();
        TFLITE_LOG(INFO) << "Openvino delegate options created" << "\n";
        }

    bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                         const TfLiteNode* node,
                                         TfLiteContext* context) const override {
        return CheckNodeSupportByOpenVINO(registration, node, context);
    }

    TfLiteStatus Initialize(TfLiteContext* context) override {
        return kTfLiteOk;
    }

    const char* Name() const override {
        return "OpenVINO SimpleDelegate";
    }

    std::unique_ptr<SimpleDelegateKernelInterface>
                    CreateDelegateKernelInterface() override {
        TFLITE_LOG(INFO) << "Creating OpenVINO delegate kernel\n";
        return std::unique_ptr<OpenVINODelegateKernel>();
    }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    auto options = SimpleDelegateInterface::Options();
    return options;
  }


    private:
        TfLiteOpenVINODelegateOptions options_;
        //std::string device_type;

}; 
}
}

TfLiteDelegate* TFL_CAPI_EXPORT TfLiteCreateOpenVINODelegate(const TfLiteOpenVINODelegateOptions* options) {
    auto ovdelegate_ = std::make_unique<tflite::openvinodelegate::OpenVINODelegate>(options);
    return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(ovdelegate_));
}

void TFL_CAPI_EXPORT TfLiteDeleteOpenVINODelegate(TfLiteDelegate* delegate) {
    return;
}

TfLiteOpenVINODelegateOptions TFL_CAPI_EXPORT TfLiteOpenVINODelegateOptionsDefault() {
    TfLiteOpenVINODelegateOptions result;
    result.debug_level = 0;
    result.plugins_path= "";
    return result;
}
