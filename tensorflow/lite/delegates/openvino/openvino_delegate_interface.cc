#include <ie_cnn_network.h>
#include "openvino/runtime/core.hpp"

#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include "tensorflow/lite/builtin_ops.h"
#include "openvino_delegate.h"

namespace tflite {
namespace openvinodelegate {
class OpenVINODelegate : public SimpleDelegateInterface {
    public:
    explicit OpenVINODelegate(TfLiteOpenVINODelegateOptions& options) {
        debug_level = options.debug_level;
        plugin_path = options.plugins_path;
        /* device_type = options.device_type; */
    }

    bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                         const TfLiteNode* node,
                                         TfLiteContext* context) const override {
        if(registration->builtin_code == kTfLiteBuiltinConv2d) {
            return true;
        }
        else {
            return false;
        }
    }

    TfLiteStatus Initialize(TfLiteContext* context) override {
        std::shared_ptr<ov::Model> mNetwork;
        ov::Core ie("/usr/local/lib64/plugins.xml");
    }

    const char* Name() const override {
        return "OpenVINO SimpleDelegate";
    }

    std::unique_ptr<SimpleDelegateKernelInterface>
                    CreateDelegateKernelInterface() override {
        return std::unique_ptr<SimpleDelegateKernelInterface>();
    }

    SimpleDelegateInterface::Options DelegateOptions() const override {
        auto options = SimpleDelegateInterface::Options();
        options.min_nodes_per_partition = 1;
        options.max_delegated_partitions = 2;
    }
    private:
        char* plugin_path;
        int debug_level;
        //std::string device_type;

};

TfLiteDelegate* TfLiteCreateOpenVINODelegate(TfLiteOpenVINODelegateOptions& options) {
    auto ovdelegate_ = std::make_unique<OpenVINODelegate>(options);
    /* auto delegate = new TfLiteDelegate();
    delegate->Prepare = &DelegatePrepare;
    delegate->flags = flag;
    delegate->CopyFromBufferHandle = nullptr;
    delegate->CopyToBufferHandle = nullptr;
    delegate->FreeBufferHandle = nullptr;
    delegate->data_ = simple_delegate.release();
    return delegate; */
}

void TFL_CAPI_EXPORT TfLiteDeleteOpenVINODelegate(TfLiteDelegate* delegate) {
    return;
}

TfLiteOpenVINODelegateOptions TfLiteOpenVINODelegateOptionsDefault() {
    TfLiteOpenVINODelegateOptions result{0};
    return result;
}
}
}