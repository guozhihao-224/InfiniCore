#include "any_cpu.h"
#include "../../../../utils.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../tensor.h"
#include <algorithm>
#include <vector>

namespace op::any::cpu {

// 辅助函数：将线性索引转换为多维索引
namespace {
inline void linear_to_multi_index(
    size_t linear_idx,
    const std::vector<size_t> &shape,
    std::vector<size_t> &indices) {
    size_t temp = linear_idx;
    for (size_t i = shape.size(); i-- > 0;) {
        indices[i] = temp % shape[i];
        temp /= shape[i];
    }
}
} // namespace

// 判断元素是否为 True（根据数据类型）
template <typename T>
struct IsTrue {
    bool operator()(const T &value) const {
        return value != T(0);
    }
};

template <>
struct IsTrue<bool> {
    bool operator()(const bool &value) const {
        return value;
    }
};

// 计算笛卡尔积迭代器（用于规约维度）
class CartesianProductIterator {
public:
    CartesianProductIterator(const std::vector<size_t> &dims)
        : dims(dims), indices(dims.size(), 0), first_call(true) {}

    bool next() {
        if (first_call) {
            first_call = false;
            return !dims.empty(); // 如果为空，直接返回 false
        }

        for (int i = dims.size() - 1; i >= 0; --i) {
            indices[i]++;
            if (indices[i] < dims[i]) {
                return true;
            }
            indices[i] = 0;
        }
        return false;
    }

    const std::vector<size_t> &get() const {
        return indices;
    }

private:
    std::vector<size_t> dims;
    std::vector<size_t> indices;
    bool first_call;
};

// 全局 any kernel
template <typename T>
void any_global_kernel(
    const T *input_data,
    bool *output_data,
    const AnyInfo &info) {

    bool found = false;
    size_t in_numel = 1;
    for (auto s : info.in_shape) {
        in_numel *= s;
    }

    for (size_t i = 0; i < in_numel; ++i) {
        std::vector<size_t> in_indices(info.in_shape.size());
        linear_to_multi_index(i, info.in_shape, in_indices);

        size_t in_offset = 0;
        for (size_t j = 0; j < in_indices.size(); ++j) {
            in_offset += in_indices[j] * info.in_strides[j];
        }

        if (IsTrue<T>{}(input_data[in_offset])) {
            found = true;
            break;
        }
    }

    output_data[0] = found;
}

// 通用 any kernel（支持多维规约、非连续张量）
template <typename T>
void any_kernel(
    const T *input_data,
    bool *output_data,
    const AnyInfo &info) {

    // 处理全局 any
    if (info.is_global) {
        any_global_kernel(input_data, output_data, info);
        return;
    }

    size_t out_numel = 1;
    for (auto s : info.out_shape) {
        out_numel *= s;
    }

    for (size_t out_idx = 0; out_idx < out_numel; ++out_idx) {
        // 1. 将 flat 输出索引转换为多维索引
        std::vector<size_t> out_indices(info.out_shape.size());
        linear_to_multi_index(out_idx, info.out_shape, out_indices);

        // 2. 构建完整的输入索引（根据 keepdim 和 reduce_dims）
        std::vector<size_t> in_indices(info.in_shape.size());
        if (info.keepdim) {
            // 输出形状和输入相同，只是 reduce_dims 处为 1
            in_indices = out_indices;
        } else {
            // 输出形状是输入去掉 reduce_dims
            size_t out_index = 0;
            for (size_t i = 0; i < info.in_shape.size(); ++i) {
                bool is_reduce_dim = false;
                for (auto rd : info.reduce_dims) {
                    if (i == static_cast<size_t>(rd)) {
                        is_reduce_dim = true;
                        break;
                    }
                }
                if (!is_reduce_dim) {
                    in_indices[i] = out_indices[out_index++];
                }
            }
        }

        // 3. 构建笛卡尔积：遍历所有 reduce_dims 的组合
        bool found = false;
        std::vector<size_t> reduce_dims_sizes;
        for (auto rd : info.reduce_dims) {
            reduce_dims_sizes.push_back(info.in_shape[rd]);
        }

        CartesianProductIterator iter(reduce_dims_sizes);
        while (iter.next()) {
            // 构建完整的输入索引
            std::vector<size_t> full_in_indices = in_indices;
            size_t reduce_index = 0;
            for (auto rd : info.reduce_dims) {
                full_in_indices[rd] = iter.get()[reduce_index++];
            }

            // 计算输入内存偏移
            size_t in_offset = 0;
            for (size_t i = 0; i < full_in_indices.size(); ++i) {
                in_offset += full_in_indices[i] * info.in_strides[i];
            }

            // 判断是否为 True（根据 dtype）
            if (IsTrue<T>{}(input_data[in_offset])) {
                found = true;
                break;
            }
        }

        // 4. 写入输出（考虑输出张量的 strides）
        size_t out_offset = 0;
        for (size_t i = 0; i < out_indices.size(); ++i) {
            out_offset += out_indices[i] * info.out_strides[i];
        }

        output_data[out_offset] = found;
    }
}

// 创建 AnyInfo
utils::Result<AnyInfo> AnyInfo::create(
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t output_desc,
    int64_t dim,
    int32_t keepdim,
    int32_t has_dim,
    const int64_t *dims,
    size_t dims_size) {

    AnyInfo info;
    info.keepdim = (keepdim != 0);
    info.dtype = input_desc->dtype();

    // 获取输入形状和 strides
    size_t ndim = input_desc->ndim();
    info.in_shape.resize(ndim);
    info.in_strides.resize(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        info.in_shape[i] = input_desc->dim(i);
        info.in_strides[i] = input_desc->stride(i);
    }

    // 处理 reduce_dims
    if (has_dim) {
        if (dims != nullptr && dims_size > 0) {
            for (size_t i = 0; i < dims_size; ++i) {
                int64_t d = dims[i];
                if (d < 0) {
                    d += static_cast<int64_t>(ndim);
                }
                if (d < 0 || d >= static_cast<int64_t>(ndim)) {
                    return utils::Result<AnyInfo>(INFINI_STATUS_BAD_TENSOR_SHAPE);
                }
                info.reduce_dims.push_back(d);
            }
            std::sort(info.reduce_dims.begin(), info.reduce_dims.end());
            info.reduce_dims.erase(
                std::unique(info.reduce_dims.begin(), info.reduce_dims.end()),
                info.reduce_dims.end());
        } else {
            int64_t d = dim;
            if (d < 0) {
                d += static_cast<int64_t>(ndim);
            }
            if (d < 0 || d >= static_cast<int64_t>(ndim)) {
                return utils::Result<AnyInfo>(INFINI_STATUS_BAD_TENSOR_SHAPE);
            }
            info.reduce_dims.push_back(d);
        }
    }

    info.is_global = info.reduce_dims.empty();

    // 获取输出形状和 strides（从 output_desc）
    size_t out_ndim = output_desc->ndim();
    info.out_shape.resize(out_ndim);
    info.out_strides.resize(out_ndim);
    for (size_t i = 0; i < out_ndim; ++i) {
        info.out_shape[i] = output_desc->dim(i);
        info.out_strides[i] = output_desc->stride(i);
    }

    return utils::Result<AnyInfo>(info);
}

// 实际计算函数（根据 dtype 分发）
infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {

    switch (_info.dtype) {
    case INFINI_DTYPE_BOOL:
        any_kernel(static_cast<const bool *>(input),
                   static_cast<bool *>(output), _info);
        break;
    case INFINI_DTYPE_U8:
        any_kernel(static_cast<const uint8_t *>(input),
                   static_cast<bool *>(output), _info);
        break;
    case INFINI_DTYPE_F32:
        any_kernel(static_cast<const float *>(input),
                   static_cast<bool *>(output), _info);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    int64_t dim,
    int32_t keepdim,
    int32_t has_dim,
    const int64_t *dims,
    size_t dims_size) {

    auto result = AnyInfo::create(
        input_desc, output_desc, dim, keepdim, has_dim, dims, dims_size);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        result.take(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::any::cpu
