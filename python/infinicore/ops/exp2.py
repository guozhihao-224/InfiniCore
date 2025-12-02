from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

def exp2(input, *, out=None):
    if out is None:
        return Tensor(_infinicore.exp2(input._underlying))
    _infinicore.exp2_(out._underlying, input._underlying)
    return out