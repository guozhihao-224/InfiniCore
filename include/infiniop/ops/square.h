#pragma once

#ifndef __INFINIOP_SQUARE_API_H_
#define __INFINIOP_SQUARE_API_H_

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSquareDescriptor_t;

__C __export infiniStatus_t infiniopCreateSquareDescriptor(
    infiniopHandle_t handle,
    infiniopSquareDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output,
    infiniopTensorDescriptor_t input);

__C __export infiniStatus_t
infiniopGetSquareWorkspaceSize(infiniopSquareDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopSquare(
    infiniopSquareDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream);

__C __export infiniStatus_t
infiniopDestroySquareDescriptor(infiniopSquareDescriptor_t desc);

#endif // __INFINIOP_SQUARE_API_H_
