#pragma once
#include <pybind11/pybind11.h>
void bindSample(pybind11::module_&);void bindSimulator(pybind11::module_&);void bindGeneration(pybind11::module_&);void bindDatasets(pybind11::module_&);void bindPreprocessing(pybind11::module_&);void bindModel(pybind11::module_&);void bindInference(pybind11::module_&);
