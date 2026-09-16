#pragma once
#include <ibeamlab/model_package.h>
#include <ibeamlab/simulator.h>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace ibeamlab::inference {
struct InferenceOptions {
    int intraOpThreads{1};
    int interOpThreads{1};
    std::string executionProvider{"cpu"};
    bool enableGraphOptimizations{false};
};
struct NamedValue { std::string name; float value{}; std::string unit; };
struct InferenceResult { std::vector<NamedValue> values; };
class OnnxInferenceEngine {
public:
    explicit OnnxInferenceEngine(model::ModelPackage package, InferenceOptions options={});
    ~OnnxInferenceEngine();
    OnnxInferenceEngine(OnnxInferenceEngine&&) noexcept;
    OnnxInferenceEngine& operator=(OnnxInferenceEngine&&) noexcept;
    std::vector<InferenceResult> predict(const std::vector<std::vector<simulator::Spectrum>>& batch) const;
private: struct Impl;std::unique_ptr<Impl> impl_;
};
}
