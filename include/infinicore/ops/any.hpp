#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <optional>
#include <vector>

namespace infinicore::op {
class Any {
public:
    using schema = void (*)(Tensor, Tensor, std::optional<int64_t>, bool);
    static void execute(Tensor output, Tensor input, 
                        std::optional<int64_t> dim, bool keepdim);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor any(Tensor input, std::optional<int64_t> dim = std::nullopt, bool keepdim = false);
Tensor any(Tensor input, const std::vector<int64_t> &dims, bool keepdim = false);
void any_(Tensor output, Tensor input, std::optional<int64_t> dim = std::nullopt, bool keepdim = false);
void any_(Tensor output, Tensor input, const std::vector<int64_t> &dims, bool keepdim = false);

} // namespace infinicore::op

