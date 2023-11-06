#include <openvino/openvino.hpp>

class IEManager {
  void infer();
  ov::InferRequest getInferRequest();
  void compile_and_serialize();
  ov::Tensor getInputBlob(const int index);
  ov::Tensor getOutputBlob(const int index);

public:
  ov::InferRequest inferRequest;
  ov::CompiledModel compiled_model;
  std::shared_ptr<ov::Model> model;
};
