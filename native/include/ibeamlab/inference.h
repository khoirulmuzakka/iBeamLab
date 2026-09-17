#pragma once
/** @file inference.h @brief Shared execution options for forward and inverse ONNX models. */
#include <string>
namespace ibeamlab::inference {
struct InferenceOptions { int intraOpThreads{1}; int interOpThreads{1}; std::string executionProvider{"cpu"}; bool enableGraphOptimizations{false}; };
}
