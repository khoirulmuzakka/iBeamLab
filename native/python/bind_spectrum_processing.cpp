#include "bindings.h"

#include <ibeamlab/spectrum_processing.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bindSpectrumProcessing(py::module_ &root) {
    auto m = root.def_submodule("spectrum");
    m.def("rebin", &ibeamlab::spectrum::rebin, py::call_guard<py::gil_scoped_release>());
    m.def("pileup", &ibeamlab::spectrum::pileup, py::arg("spectrum"), py::arg("real_time"),
          py::arg("live_time"), py::arg("fudge_factor"), py::arg("clip_negative") = true,
          py::call_guard<py::gil_scoped_release>());
    m.def("energy_to_channel_and_pileup", &ibeamlab::spectrum::energyToChannelAndPileup,
          py::arg("energy_spectrum"), py::arg("calibration_offset"),
          py::arg("calibration_linear"), py::arg("calibration_quadratic"),
          py::arg("real_time"), py::arg("live_time"), py::arg("fudge_factor"),
          py::arg("scale") = 1.0, py::arg("clip_negative") = true,
          py::call_guard<py::gil_scoped_release>());
    m.def("crop_or_pad", &ibeamlab::spectrum::cropOrPad, py::arg("spectrum"),
          py::arg("size"), py::arg("padding") = 0.0,
          py::call_guard<py::gil_scoped_release>());
    m.def("concatenate", &ibeamlab::spectrum::concatenate,
          py::call_guard<py::gil_scoped_release>());
    m.def("clip", &ibeamlab::spectrum::clip, py::call_guard<py::gil_scoped_release>());
}
