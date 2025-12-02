#pragma once

#include "infinicore/ops/exp2.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {
inline void bind_exp2(py::module &m) {
    m.def("exp2", &op::exp2, py::arg("input"),
          R"doc(Element-wise 2^x.)doc");
    m.def("exp2_", &op::exp2_, py::arg("output"), py::arg("input"),
          R"doc(In-place 2^x.)doc");
}
} // namespace infinicore::ops