#include "bindings.h"
#include <ibeamlab/transforms.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace ibeamlab::preprocessing;

namespace {
Matrix matrixFromArray(const py::array_t<float, py::array::c_style | py::array::forcecast> &input) {
    if (input.ndim() != 2)
        throw py::value_error("expected a two-dimensional float32 array");
    Matrix result(static_cast<std::size_t>(input.shape(0)),
                  std::vector<float>(static_cast<std::size_t>(input.shape(1))));
    auto view = input.unchecked<2>();
    for (py::ssize_t r = 0; r < input.shape(0); ++r)
        for (py::ssize_t c = 0; c < input.shape(1); ++c)
            result[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = view(r, c);
    return result;
}
py::array_t<float> arrayFromMatrix(const Matrix &input) {
    const auto rows = input.size(), cols = rows ? input.front().size() : 0;
    py::array_t<float> result({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
    auto view = result.mutable_unchecked<2>();
    for (std::size_t r = 0; r < rows; ++r) {
        if (input[r].size() != cols)
            throw std::runtime_error("ragged native matrix");
        for (std::size_t c = 0; c < cols; ++c)
            view(r, c) = input[r][c];
    }
    return result;
}
py::array_t<float> apply(const Transform &self,
                         const py::array_t<float, py::array::c_style | py::array::forcecast> &input,
                         bool inverse) {
    auto matrix = matrixFromArray(input);
    Matrix output;
    {
        py::gil_scoped_release release;
        output = inverse ? self.inverse(matrix) : self.apply(matrix);
    }
    return arrayFromMatrix(output);
}
template <class T> void common(py::class_<T, Transform, std::shared_ptr<T>> &c) {
    using Array = py::array_t<float, py::array::c_style | py::array::forcecast>;
    c.def("apply", [](const T &self, const Array &input) { return apply(self, input, false); })
        .def("inverse", [](const T &self, const Array &input) { return apply(self, input, true); })
        .def_property_readonly("input_dimension", &T::inputDimension);
}
} // namespace

void bindTransforms(py::module_ &root) {
    auto m = root.def_submodule("transforms");
    py::class_<Transform, std::shared_ptr<Transform>>(m, "Transform");
    py::class_<IdentityTransform, Transform, std::shared_ptr<IdentityTransform>> identity(
        m, "IdentityTransform");
    common(identity);
    identity.def(py::init<std::size_t>());
    py::class_<ConstantFactorTransform, Transform, std::shared_ptr<ConstantFactorTransform>>
        constant(m, "ConstantFactorTransform");
    common(constant);
    constant.def(py::init<std::size_t, float>());
    py::class_<StandardScaler, Transform, std::shared_ptr<StandardScaler>> standard(
        m, "StandardScaler");
    common(standard);
    standard.def(py::init<std::vector<float>, std::vector<float>>());
    py::class_<MinMaxScaler, Transform, std::shared_ptr<MinMaxScaler>> minmax(m, "MinMaxScaler");
    common(minmax);
    minmax.def(py::init<std::vector<float>, std::vector<float>, float, float>(), py::arg("minimum"),
               py::arg("scale"), py::arg("low") = 0.0F, py::arg("high") = 1.0F);
    py::class_<LogTransform, Transform, std::shared_ptr<LogTransform>> logarithm(m, "LogTransform");
    common(logarithm);
    logarithm.def(py::init<std::size_t, float>(), py::arg("dimension"), py::arg("offset") = 1.0F);
    py::class_<TransformPipeline, Transform, std::shared_ptr<TransformPipeline>> pipeline(
        m, "TransformPipeline");
    common(pipeline);
    pipeline.def(py::init<>()).def("add", &TransformPipeline::add);
}
