from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def square(input, *, out=None):
    if out is None:
        return Tensor(_infinicore.square(input._underlying))

    _infinicore.square_(out._underlying, input._underlying)

    return out