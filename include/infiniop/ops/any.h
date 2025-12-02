#pragma once

#include "../operator_descriptor.h"
#include <cstdint>

typedef struct InfiniopDescriptor *infiniopAnyDescriptor_t;

__C __export infiniStatus_t infiniopCreateAnyDescriptor(
    infiniopHandle_t handle,
    infiniopAnyDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    int64_t dim,
    int32_t keepdim,
    int32_t has_dim,
    const int64_t *dims,
    size_t dims_size);

__C __export infiniStatus_t
infiniopGetAnyWorkspaceSize(infiniopAnyDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopAny(
    infiniopAnyDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream);

__C __export infiniStatus_t
infiniopDestroyAnyDescriptor(infiniopAnyDescriptor_t desc);
