#include "infinicore/ops/any.hpp"

namespace infinicore::op {

common::OpDispatcher<Any::schema> &Any::dispatcher() {
    static common::OpDispatcher<Any::schema> dispatcher_;
    return dispatcher_;
}

void Any::execute(Tensor output, Tensor input,
                  std::optional<int64_t> dim, bool keepdim) {
    dispatcher().lookup(context::getDevice().getType())(
        output, input, dim, keepdim);
}

Tensor any(Tensor input, std::optional<int64_t> dim, bool keepdim) {
    // 计算输出形状
    auto in_shape = input->shape();
    std::vector<size_t> out_shape;

    if (!dim.has_value()) {
        // 全局 any: 输出为标量
        out_shape = {};
    } else {
        int64_t d = dim.value();
        if (d < 0) {
            d += static_cast<int64_t>(in_shape.size());
        }
        out_shape = in_shape;
        if (keepdim) {
            out_shape[d] = 1;
        } else {
            out_shape.erase(out_shape.begin() + d);
        }
    }

    auto out = Tensor::empty(out_shape, DataType::BOOL, input->device());
    any_(out, input, dim, keepdim);
    return out;
}

Tensor any(Tensor input, const std::vector<int64_t> &dims, bool keepdim) {
    auto in_shape = input->shape();
    std::vector<size_t> out_shape;

    // 规范化维度
    std::vector<int64_t> norm_dims = dims;
    for (auto &d : norm_dims) {
        if (d < 0) {
            d += static_cast<int64_t>(in_shape.size());
        }
    }
    std::sort(norm_dims.begin(), norm_dims.end());
    norm_dims.erase(
        std::unique(norm_dims.begin(), norm_dims.end()),
        norm_dims.end());

    if (keepdim) {
        out_shape = in_shape;
        for (int64_t d : norm_dims) {
            out_shape[d] = 1;
        }
    } else {
        std::vector<bool> keep(in_shape.size(), true);
        for (int64_t d : norm_dims) {
            keep[d] = false;
        }
        for (size_t i = 0; i < in_shape.size(); ++i) {
            if (keep[i]) {
                out_shape.push_back(in_shape[i]);
            }
        }
    }

    auto out = Tensor::empty(out_shape, DataType::BOOL, input->device());
    any_(out, input, dims, keepdim);
    return out;
}

void any_(Tensor output, Tensor input, std::optional<int64_t> dim, bool keepdim) {
    Any::execute(output, input, dim, keepdim);
}

void any_(Tensor output, Tensor input, const std::vector<int64_t> &dims, bool keepdim) {
    // 对于多个维度，需要特殊处理
    // 这里简化处理，实际应该调用支持多维度的版本
    if (dims.size() == 1) {
        any_(output, input, dims[0], keepdim);
    } else {
        // 多个维度：递归 reduce
        Tensor temp = input;
        for (size_t i = 0; i < dims.size(); ++i) {
            bool last = (i == dims.size() - 1);
            temp = any(temp, dims[i], last ? keepdim : true);
        }
        output->copy_from(temp);
    }
}

} // namespace infinicore::op
