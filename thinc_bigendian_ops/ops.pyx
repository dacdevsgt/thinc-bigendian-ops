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
        print("in big-endian-ops asarray")
        if isinstance(data, self.xp.ndarray):
            if dtype is not None:
                if data.dtype.byteorder == "<":
                    print("swapping byte order, was: ",data.dtype.byteorder)
                    data = data.byteswap().newbyteorder()
                else
                    print("no swap, was  ",data.dtype.byteorder)
                return self.xp.asarray(data, dtype=dtype)
            else:
                print("no swap, was none")
                return self.xp.asarray(data)
        elif hasattr(data, 'numpy'):
            # Handles PyTorch Tensor
            print("no swap, was torch")
            return data.numpy()
        elif hasattr(data, "get"):
            print("no swap, was get")
            return data.get()
        elif dtype is not None:
            print("no swap, dtype is not none")
            return self.xp.array(data, dtype=dtype)
        else:
            print("no swap, else")
            return self.xp.array(data)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def hash(self, const uint64_t[::1] ids, uint32_t seed):
        """Hash a sequence of 64-bit keys into a table with 4 32-bit keys."""
        # Written to mirror the GPU implementation
        cdef np.ndarray[uint32_t, ndim=2] keys = self.alloc((ids.shape[0], 4), dtype='uint32')
        cdef int i, j
        cdef unsigned char entropy[16] # 128/8=16
        cdef size_t n_items = len(ids)
        cdef size_t in_size = sizeof(uint64_t)
        cdef unsigned char src[16] 
        dest = <unsigned char*>keys.data
        # byteorder (endian) compatibility change:
        #   numpy_ops hash maps src as a char array over input uint64_t id values. Each byte (char)
        #   in each uint64 value is operated on by the underlying hash logic. However, this method of 
        #   mapping the hash results a reversal of the resulting keys on big endian platforms. The
        #   approach used here implements a byteorder agnostic approach that will result in an 
        #   consistent result regardless of implementation platform. 
        print("in big-endian-ops hash")
        for i in range(n_items):
            for k in range(in_size):
                src[k] = (ids[i] >> (k*8)) & 0xFF
            hash128_x64(<void*>src, in_size, seed, entropy)
            for j in range(16):
                dest[j] = entropy[j]
            dest += 16
        return keys 
