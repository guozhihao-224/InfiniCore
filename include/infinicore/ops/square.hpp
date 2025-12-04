#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Square {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor output, Tensor input);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor square(Tensor input);
} // namespace infinicore::op