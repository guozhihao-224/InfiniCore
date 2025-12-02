#include "infinicore/ops/exp2.hpp"

namespace infinicore::op {

common::OpDispatcher<Exp2::schema> &Exp2::dispatcher() {
    static common::OpDispatcher<Exp2::schema> dispatcher_;
    return dispatcher_;
}

void Exp2::execute(Tensor output, Tensor input) {
    dispatcher().lookup(context::getDevice().getType())(output, input);
}

Tensor exp2(Tensor input) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    exp2_(output, input);
    return output;
}

void exp2_(Tensor output, Tensor input) {
    Exp2::execute(output, input);
}

} // namespace infinicore::op