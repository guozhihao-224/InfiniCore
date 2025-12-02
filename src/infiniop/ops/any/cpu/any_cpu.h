#ifndef __ANY_CPU_H__
#define __ANY_CPU_H__

#include "../../../../utils.h"
#include "../../../operator.h"
#include <vector>

namespace op::any::cpu {

struct AnyInfo {
    std::vector<size_t> in_shape;
    std::vector<ptrdiff_t> in_strides;
    std::vector<size_t> out_shape;
    std::vector<ptrdiff_t> out_strides;
    std::vector<int64_t> reduce_dims;
    bool keepdim;
    bool is_global;
    infiniDtype_t dtype;

    static utils::Result<AnyInfo> create(
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t output_desc,
        int64_t dim,
        int32_t keepdim,
        int32_t has_dim,
        const int64_t *dims,
        size_t dims_size);
};

class Descriptor : public InfiniopDescriptor {
public:
    AnyInfo _info;

    Descriptor(AnyInfo info, infiniDevice_t device, int device_id)
        : InfiniopDescriptor{device, device_id}, _info(std::move(info)) {}

    ~Descriptor() = default;

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc,
        int64_t dim,
        int32_t keepdim,
        int32_t has_dim,
        const int64_t *dims,
        size_t dims_size);

    size_t workspaceSize() const { return 0; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *output,
        const void *input,
        void *stream) const;
};

} // namespace op::any::cpu

#endif // __ANY_CPU_H__
