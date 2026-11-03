// Standalone runtime probe: no Python and no external CUDA DLL search paths.
#include <onnxruntime_cxx_api.h>
#include <ibeamlab/inference/options.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: ibeamlab_runtime_probe model.onnx CPU|CUDA");
        Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "ibeamlab-runtime-probe");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        const std::string provider = argv[2];
        if (provider != "CPU" && provider != "CUDA") throw std::runtime_error("invalid provider");
        const auto providers = ibeamlab::inference::available_execution_providers();
        const std::string native_provider = provider == "CUDA" ? "cuda" : "cpu";
        if (std::find(providers.begin(), providers.end(), native_provider) == providers.end())
            throw std::runtime_error("requested provider is unavailable in the iBeamLab build");
        if (provider == "CUDA") {
            OrtCUDAProviderOptions cuda{};
            options.AppendExecutionProvider_CUDA(cuda);
            options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
        }
        std::ifstream stream(argv[1], std::ios::binary);
        if (!stream) throw std::runtime_error("probe model not found");
        const std::vector<char> graph((std::istreambuf_iterator<char>(stream)), {});
        Ort::Session session(environment, graph.data(), graph.size(), options);
        if (session.GetInputCount() != 1 || session.GetOutputCount() != 1)
            throw std::runtime_error("probe expects one input and output");
        auto info = session.GetInputTypeInfo(0);
        auto shape = info.GetTensorTypeAndShapeInfo().GetShape();
        std::size_t size = 1;
        for (auto &dimension : shape) {
            if (dimension < 1) dimension = 1;
            size *= static_cast<std::size_t>(dimension);
        }
        std::vector<float> values(size, 1.f);
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input = Ort::Value::CreateTensor<float>(memory, values.data(), values.size(), shape.data(), shape.size());
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = session.GetInputNameAllocated(0, allocator);
        auto output_name = session.GetOutputNameAllocated(0, allocator);
        const char *inputs[]{input_name.get()}, *outputs[]{output_name.get()};
        auto result = session.Run(Ort::RunOptions{nullptr}, inputs, &input, 1, outputs, 1);
        if (!result[0].IsTensor()) throw std::runtime_error("probe returned no tensor");
        const auto count = result[0].GetTensorTypeAndShapeInfo().GetElementCount();
        const auto *output = result[0].GetTensorData<float>();
        double sum = 0;
        for (std::size_t i = 0; i < count; ++i) {
            if (!std::isfinite(output[i])) throw std::runtime_error("nonfinite probe result");
            sum += output[i];
        }
        std::cout << "{\"output_count\":" << count << ",\"output_sum\":" << sum
                  << ",\"provider\":\"" << provider << "\",\"onnxruntime_version\":\""
                  << OrtGetApiBase()->GetVersionString() << "\"}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
