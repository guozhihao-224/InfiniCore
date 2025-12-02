#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/any.h"

#ifdef ENABLE_CPU_API
#include "cpu/any_cpu.h"
#endif

__C infiniStatus_t infiniopCreateAnyDescriptor(
    infiniopHandle_t handle,
    infiniopAnyDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    int64_t dim,
    int32_t keepdim,
    int32_t has_dim,
    const int64_t *dims,
    size_t dims_size) {

#define CREATE(CASE, NAMESPACE)                                            \
    case CASE:                                                             \
        return op::any::NAMESPACE::Descriptor::create(                     \
            handle,                                                        \
            reinterpret_cast<op::any::NAMESPACE::Descriptor **>(desc_ptr), \
            output_desc, input_desc, dim, keepdim, has_dim, dims, dims_size)

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopGetAnyWorkspaceSize(infiniopAnyDescriptor_t desc,
                                               size_t *size) {

#define GET(CASE, NAMESPACE)                                             \
    case CASE:                                                           \
        *size = reinterpret_cast<op::any::NAMESPACE::Descriptor *>(desc) \
                    ->workspaceSize();                                   \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopAny(infiniopAnyDescriptor_t desc, void *workspace,
                               size_t workspace_size, void *output,
                               const void *input, void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                            \
    case CASE:                                                                \
        return reinterpret_cast<const op::any::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, output, input, stream)

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t infiniopDestroyAnyDescriptor(infiniopAnyDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                \
    case CASE:                                                                 \
        delete reinterpret_cast<const op::any::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
