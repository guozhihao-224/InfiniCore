#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/exp2.hpp"
#include "infiniop/ops/exp2.h"
#include <infiniop.h>

namespace infinicore::op::exp2_impl::infiniop {

thread_local common::OpCache<size_t, infiniopExp2Descriptor_t> caches(
    100,
    [](infiniopExp2Descriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyExp2Descriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input) {
    size_t seed = hash_combine(output, input);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();
    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopExp2Descriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateExp2Descriptor(
            context::getInfiniopHandle(output->device()),
            &desc,
            output->desc(),
            input->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetExp2WorkspaceSize(desc, &workspace_size));

    std::shared_ptr<Memory> workspace;
    void *workspace_ptr = nullptr;
    if (workspace_size > 0) {
        workspace = context::allocateMemory(workspace_size);
        workspace_ptr = workspace->data();
    }

    INFINICORE_CHECK_ERROR(infiniopExp2(
        desc,
        workspace_ptr,
        workspace_size,
        output->data(),
        input->data(),
        context::getStream()));
}

static bool registered = []() {
    Exp2::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::exp2_impl::infiniop