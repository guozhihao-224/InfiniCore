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
    if (dims.size() == 1) {
        any_(output, input, dims[0], keepdim);
    } else {
        // 多个维度：从大到小排序，从后往前 reduce
        std::vector<int64_t> sorted_dims = dims;
        std::sort(sorted_dims.rbegin(), sorted_dims.rend()); // 降序排序
        
        Tensor temp = input;
        // 先全部用 keepdim=True 进行 reduce
        for (int64_t d : sorted_dims) {
            temp = any(temp, d, true);
        }
        
        // 如果最终不需要 keepdim，需要重新计算（因为需要移除维度）
        if (!keepdim && temp->shape() != output->shape()) {
            // 重新计算，这次正确处理 keepdim
            temp = input;
            for (size_t i = 0; i < sorted_dims.size(); ++i) {
                bool last = (i == sorted_dims.size() - 1);
                temp = any(temp, sorted_dims[i], last ? keepdim : true);
            }
        }
        
        output->copy_from(temp);
    }
}

} // namespace infinicore::op
