cimport cython
cimport numpy as np

from libc.stdint cimport uint32_t, uint64_t
from typing import Optional
import numpy
from thinc.api import NumpyOps
from thinc.config import registry
from murmurhash.mrmr cimport hash64, hash128_x86, hash128_x64

@registry.ops("BigEndianOps")    
class BigEndianOps(NumpyOps):
    """Thinc Ops class that handles big endian impacts for some
    operations. Other operations fall back to numpy."""
    name = "bigendian"
    xp = numpy

    def asarray(self, data, dtype=None):
        # If we detect little endian data, we should byteswap and correct byteorder 
        if isinstance(data, self.xp.ndarray):
            if dtype is not None:
                out = self.xp.asarray(data, dtype=dtype)
            else:
                out = self.xp.asarray(data)
        elif hasattr(data, 'numpy'):
            # Handles PyTorch Tensor
            out = data.numpy()
        elif hasattr(data, "get"):
            out = data.get()
        elif dtype is not None:
            out = self.xp.array(data, dtype=dtype)
        else:
            out = self.xp.array(data)
        
        if out.dtype.byteorder == "<":
            return out.byteswap().view(out.dtype.newbyteorder())
        else:
            return out
