#pragma once

#include "infinicore/ops/any.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace infinicore::ops {

Tensor py_any(Tensor input, py::object dim, bool keepdim) {
    if (dim.is_none()) {
        return op::any(input, std::nullopt, keepdim);
    } else if (py::isinstance<py::int_>(dim)) {
        return op::any(input, dim.cast<int64_t>(), keepdim);
    } else if (py::isinstance<py::tuple>(dim) || py::isinstance<py::list>(dim)) {
        auto dims = dim.cast<std::vector<int64_t>>();
        return op::any(input, dims, keepdim);
    } else {
        throw std::runtime_error("dim must be None, int, or tuple/list");
    }
}

void py_any_(Tensor output, Tensor input, py::object dim, bool keepdim) {
    if (dim.is_none()) {
        op::any_(output, input, std::nullopt, keepdim);
    } else if (py::isinstance<py::int_>(dim)) {
        op::any_(output, input, dim.cast<int64_t>(), keepdim);
    } else if (py::isinstance<py::tuple>(dim) || py::isinstance<py::list>(dim)) {
        auto dims = dim.cast<std::vector<int64_t>>();
        op::any_(output, input, dims, keepdim);
    } else {
        throw std::runtime_error("dim must be None, int, or tuple/list");
    }
}

inline void bind_any(py::module &m) {
    m.def("any", &py_any,
          py::arg("input"),
          py::arg("dim") = py::none(),
          py::arg("keepdim") = false,
          R"doc(Returns True if any element in the input tensor is True.)doc");

    m.def("any_", &py_any_,
          py::arg("output"),
          py::arg("input"),
          py::arg("dim") = py::none(),
          py::arg("keepdim") = false,
          R"doc(In-place version of any.)doc");
}

} // namespace infinicore::ops
