#pragma once
#include <ibeamlab/sample.h>
#include <string>
namespace ibeamlab::sample {
std::string toToml(const SampleModel& sample);
SampleModel sampleModelFromToml(const std::string& text);
std::string toToml(const ExperimentalSetup& setup);
ExperimentalSetup experimentalSetupFromToml(const std::string& text);
}
