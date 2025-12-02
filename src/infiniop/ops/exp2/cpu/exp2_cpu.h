#ifndef __EXP2_CPU_H__
#define __EXP2_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>

ELEMENTWISE_DESCRIPTOR(exp2, cpu)

namespace op::exp2::cpu {
typedef struct Exp2Op {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        return std::exp2(x);
    }
} Exp2Op;

} // namespace op::exp2::cpu

#endif // __INFINIOP_EXP2_CPU_H__