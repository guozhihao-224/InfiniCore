#pragma once

#include "infinicore.h"
#include "infiniop/handle.h"
#include "infiniop/tensor_descriptor.h"
#ifndef __INFINIOP_EXP2_API_H_
#define __INFINIOP_EXP2_API_H_

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopExp2Descriptor_t;

__C __export infiniStatus_t infiniopCreateExp2Descriptor(
    infiniopHandle_t handle,
    infiniopExp2Descriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output,
    infiniopTensorDescriptor_t input);

__C __export infiniStatus_t
infiniopGetExp2WorkspaceSize(infiniopExp2Descriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopExp2(
    infiniopExp2Descriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream);

__C __export infiniStatus_t
infiniopDestroyExp2Descriptor(infiniopExp2Descriptor_t desc);

#endif // __INFINIOP_EXP2_API_H_