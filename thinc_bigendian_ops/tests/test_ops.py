import pytest
import numpy
from thinc_bigendian_ops.bigendian_ops import BigEndianOps


def test_endian_conversion():
    ops = BigEndianOps()
    ndarr_tst = numpy.ndarray((1, 5), dtype="<f")
    out = ops.asarray(ndarr_tst)
    assert out.dtype.byteorder == ">"

#add hash test


