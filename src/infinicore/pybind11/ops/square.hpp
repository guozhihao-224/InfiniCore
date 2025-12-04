#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/square.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_square(py::module &m) {
    m.def("square",
          &op::square,
          py::arg("input"),
          R"doc(Element-wise square: output = input * input.)doc");

    m.def("square_",
          &op::square_,
          py::arg("output"),
          py::arg("input"),
          R"doc(In-place element-wise square.)doc");
}

} // namespace infinicore::ops