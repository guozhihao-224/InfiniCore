#include "infinicore/ops/square.hpp"
#include "infinicore/context/context.hpp"

namespace infinicore::op {

common::OpDispatcher<Square::schema> &Square::dispatcher() {
    static common::OpDispatcher<Square::schema> dispatcher_;
    return dispatcher_;
};

void Square::execute(Tensor output, Tensor input) {
    dispatcher().lookup(context::getDevice().getType())(output, input);
}

void square_(Tensor output, Tensor input) {
    Square::execute(output, input);
}

Tensor square(Tensor input) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    square_(output, input);
    return output;
}

} // namespace infinicore::op