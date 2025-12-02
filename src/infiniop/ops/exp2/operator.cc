#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/exp2.h"

#ifdef ENABLE_CPU_API
#include "cpu/exp2_cpu.h"
#endif

__C infiniStatus_t infiniopCreateExp2Descriptor(
    infiniopHandle_t handle, infiniopExp2Descriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc) {

#define CREATE(CASE, NAMESPACE)                                                \
  case CASE:                                                                   \
    return op::exp2::NAMESPACE::Descriptor::create(                            \
        handle,                                                                \
        reinterpret_cast<op::exp2::NAMESPACE::Descriptor **>(desc_ptr),        \
        output_desc, {input_desc})

  switch (handle->device) {

#ifdef ENABLE_CPU_API
    CREATE(INFINI_DEVICE_CPU, cpu);
#endif

  default:
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
  }

#undef CREATE
}

__C infiniStatus_t infiniopGetExp2WorkspaceSize(infiniopExp2Descriptor_t desc,
                                                size_t *size) {

#define GET(CASE, NAMESPACE)                                                   \
  case CASE:                                                                   \
    *size = reinterpret_cast<op::exp2::NAMESPACE::Descriptor *>(desc)          \
                ->workspaceSize();                                             \
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

__C infiniStatus_t infiniopExp2(infiniopExp2Descriptor_t desc, void *workspace,
                                size_t workspace_size, void *output,
                                const void *input, void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                             \
  case CASE:                                                                   \
    return reinterpret_cast<const op::exp2::NAMESPACE::Descriptor *>(desc)     \
        ->calculate(workspace, workspace_size, output, {input}, stream)

  switch (desc->device_type) {

#ifdef ENABLE_CPU_API
    CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif

  default:
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
  }

#undef CALCULATE
}

__C infiniStatus_t
infiniopDestroyExp2Descriptor(infiniopExp2Descriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                \
  case CASE:                                                                   \
    delete reinterpret_cast<const op::exp2::NAMESPACE::Descriptor *>(desc);    \
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