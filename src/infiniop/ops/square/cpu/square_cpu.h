#ifndef __SQUARE_CPU_H__
#define __SQUARE_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(square, cpu)

namespace op::square::cpu {
typedef struct SquareOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        return x * x;
    }
} SquareOp;

} // namespace op::square::cpu

#endif // __SQUARE_CPU_H__