from algorithm.functional import vectorize
from gpu import syncwarp, warp
from gpu.id import block_dim, block_idx, thread_idx
from gpu.host import DeviceContext
from os import Atomic
from math import ceildiv
from sys.info import simd_width_of
from time.time import perf_counter


from aoclib.parse import span_to_int
from extramojo.io.buffered import BufferedReader
from extramojo.bstr.bstr import SplitIterator

def main():
    var fh = open("./small.txt", "r")
    var reader = BufferedReader(fh^)
    var buffer = List[UInt8](capacity=1024)
    var side_a: List[Int32] = []
    var side_b: List[Int32] = []
    while reader.read_until(buffer, char=UInt(ord("\n"))):
        var iter = SplitIterator(buffer, ord(" "))
        var a = span_to_int(iter.__next__())
        _ = iter.__next__()
        _ = iter.__next__()
        var b = span_to_int(iter.__next__())
        side_a.append(a)
        side_b.append(b)
        buffer.clear()
