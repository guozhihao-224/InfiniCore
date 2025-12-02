#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/any.hpp"
#include "infiniop/ops/any.h"
#include <infiniop.h>

namespace infinicore::op::any_impl::infiniop {

thread_local common::OpCache<size_t, infiniopAnyDescriptor_t> caches(
    100,
    [](infiniopAnyDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyAnyDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input, 
               std::optional<int64_t> dim, bool keepdim) {
    size_t seed = hash_combine(output, input, dim.has_value() ? dim.value() : -1, keepdim);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();
    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopAnyDescriptor_t desc = nullptr;

    if (!desc_opt) {
        int32_t has_dim = dim.has_value() ? 1 : 0;
        int64_t dim_val = dim.has_value() ? dim.value() : 0;
        
        INFINICORE_CHECK_ERROR(infiniopCreateAnyDescriptor(
            context::getInfiniopHandle(output->device()),
            &desc,
            output->desc(),
            input->desc(),
            dim_val,
            keepdim ? 1 : 0,
            has_dim,
            nullptr,  // dims
            0));      // dims_size
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetAnyWorkspaceSize(desc, &workspace_size));

    std::shared_ptr<Memory> workspace;
    void *workspace_ptr = nullptr;
    if (workspace_size > 0) {
        workspace = context::allocateMemory(workspace_size);
        workspace_ptr = workspace->data();
    }

    INFINICORE_CHECK_ERROR(infiniopAny(
        desc,
        workspace_ptr,
        workspace_size,
        output->data(),
        input->data(),
        context::getStream()));
}

static bool registered = []() {
    Any::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::any_impl::infiniop

