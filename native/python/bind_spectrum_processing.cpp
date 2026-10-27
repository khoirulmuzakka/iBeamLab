#include "bindings.h"

#include <ibeamlab/spectrum_processing.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>

namespace py = pybind11;

void bindSpectrumProcessing(py::module_ &root) {
    auto m = root.def_submodule("spectrum");
    m.def("rebin", &ibeamlab::spectrum::rebin, py::call_guard<py::gil_scoped_release>());
    m.def("pileup", &ibeamlab::spectrum::pileup, py::arg("spectrum"), py::arg("real_time"),
          py::arg("live_time"), py::arg("fudge_factor"), py::arg("clip_negative") = true,
          py::call_guard<py::gil_scoped_release>());
    // One NumPy batch crosses the Python boundary; all row work runs without the GIL.
    m.def("pileup_batch", [](const py::array_t<float, py::array::c_style | py::array::forcecast> &input,
                             double realTime, double liveTime, double fudgeFactor,
                             int threads) {
        if (input.ndim() != 2 || input.shape(1) < 1)
            throw py::value_error("pileup_batch requires a nonempty channel dimension");
        if (threads < 1)
            throw py::value_error("pileup threads must be positive");
        if (!std::isfinite(realTime) || realTime <= 0 || !std::isfinite(liveTime) || liveTime < 0 ||
            !std::isfinite(fudgeFactor) || fudgeFactor < 0)
            throw py::value_error("invalid pileup timing or fudge factor");
        const auto rows = input.shape(0), width = input.shape(1);
        const auto outputWidth = 2 * width - 1;
        py::array_t<float> output({rows, outputWidth});
        const float *source = input.data();
        float *destination = output.mutable_data();
        std::vector<std::exception_ptr> errors(static_cast<std::size_t>(rows));
        [[maybe_unused]] const int workers = static_cast<int>(std::max<py::ssize_t>(1, std::min<py::ssize_t>(threads, rows)));
        {
            py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(workers) if (workers > 1)
#endif
            for (std::int64_t row = 0; row < rows; ++row) {
                try {
                    const float *first = source + row * width;
                    std::vector<double> counts(first, first + width);
                    for (double value : counts)
                        if (!std::isfinite(value) || value < 0)
                            throw std::invalid_argument("pileup counts must be finite and nonnegative");
                    const auto piled = ibeamlab::spectrum::pileup(counts, realTime, liveTime, fudgeFactor);
                    for (py::ssize_t channel = 0; channel < outputWidth; ++channel) {
                        const double value = piled[static_cast<std::size_t>(channel)];
                        if (!std::isfinite(value) || value > std::numeric_limits<float>::max())
                            throw std::invalid_argument("pileup output is not representable as float32");
                        destination[row * outputWidth + channel] = static_cast<float>(value);
                    }
                } catch (...) {
                    errors[static_cast<std::size_t>(row)] = std::current_exception();
                }
            }
        }
        for (const auto &error : errors)
            if (error)
                std::rethrow_exception(error);
        return output;
    }, py::arg("spectra"), py::arg("real_time"), py::arg("live_time"),
       py::arg("fudge_factor"), py::arg("threads") = 1);
    m.def("energy_to_channel_and_pileup", &ibeamlab::spectrum::energyToChannelAndPileup,
          py::arg("energy_spectrum"), py::arg("calibration_offset"), py::arg("calibration_linear"),
          py::arg("calibration_quadratic"), py::arg("real_time"), py::arg("live_time"),
          py::arg("fudge_factor"), py::arg("scale") = 1.0, py::arg("clip_negative") = true,
          py::call_guard<py::gil_scoped_release>());
    m.def("remove_pileup", &ibeamlab::spectrum::removePileup, py::arg("spectrum"),
          py::arg("real_time"), py::arg("live_time"), py::arg("fudge_factor"),
          py::arg("relative_tolerance") = 1e-6, py::call_guard<py::gil_scoped_release>());
    m.def("crop_or_pad", &ibeamlab::spectrum::cropOrPad, py::arg("spectrum"), py::arg("size"),
          py::arg("padding") = 0.0, py::call_guard<py::gil_scoped_release>());
    m.def("concatenate", &ibeamlab::spectrum::concatenate,
          py::call_guard<py::gil_scoped_release>());
    m.def("clip", &ibeamlab::spectrum::clip, py::call_guard<py::gil_scoped_release>());
}
