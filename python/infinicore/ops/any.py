from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

def any(input, dim=None, keepdim=False, *, out=None):
    if out is None:
        return Tensor(_infinicore.any(input._underlying, dim, keepdim))
    _infinicore.any_(out._underlying, input._underlying, dim, keepdim)
    return out

