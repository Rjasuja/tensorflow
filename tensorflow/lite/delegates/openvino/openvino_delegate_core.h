#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_DELEGATE_CORE_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_DELEGATE_CORE_H_
#include <iostream>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/serialize.hpp>
#include <openvino/runtime/core.hpp>
#include "openvino_graph_builder.h"
namespace tflite {
namespace openvinodelegate {
class OpenVINODelegateManager {
public:
    OpenVINODelegateManager(std::string_view plugins_path) : openvino_delegate_core(ov::Core()) {
        plugins_location = plugins_path;
    }
    TfLiteStatus openvino_delegate_init() {
        std::vector<std::string> ovDevices = openvino_delegate_core.get_available_devices();
        if (std::find(ovDevices.begin(), ovDevices.end(), "CPU") == ovDevices.end()) {
            return kTfLiteDelegateError;
        } else {
            return kTfLiteOk;
        }
    }

    std::vector<int> getComputeInputs() { return compute_inputs; }

    std::vector<int> getOutputs() { return outputs; }

    TfLiteStatus createGraphfromTfLite(TfLiteContext* context, const TfLiteDelegateParams* params);
    ov::InferRequest inferRequest;

private:
    std::unique_ptr<OpenVINOGraphBuilder> openvino_graph_builder;
    ov::Core openvino_delegate_core;
    std::string plugins_location;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    std::string deviceStr = "CPU";
    std::vector<int> compute_inputs = {};
    std::vector<int> outputs = {};
};
}  // namespace openvinodelegate
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_OPENVINO_DELEGATE_KERNEL_H_
