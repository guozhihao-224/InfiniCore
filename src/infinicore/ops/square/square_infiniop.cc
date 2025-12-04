#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/square.hpp"
#include "infiniop/ops/square.h"
#include <infiniop.h>

namespace infinicore::op::square_impl::infiniop {

thread_local common::OpCache<size_t, infiniopSquareDescriptor_t> caches(
    100, // capacity
    [](infiniopSquareDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroySquareDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input) {
    size_t seed = hash_combine(output, input);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopSquareDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateSquareDescriptor(
            context::getInfiniopHandle(output->device()),
            &desc,
            output->desc(),
            input->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetSquareWorkspaceSize(desc, &workspace_size));

    std::shared_ptr<Memory> workspace;
    void *workspace_ptr = nullptr;
    if (workspace_size > 0) {
        workspace = context::allocateMemory(workspace_size);
        workspace_ptr = workspace->data();
    }

    INFINICORE_CHECK_ERROR(infiniopSquare(
        desc,
        workspace_ptr,
        workspace_size,
        output->data(),
        input->data(),
        context::getStream()));
}

static bool registered = []() {
    Square::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::square_impl::infiniop