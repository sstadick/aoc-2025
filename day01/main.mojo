from algorithm.functional import vectorize
from gpu import syncwarp, warp
from gpu.id import block_dim, block_idx, thread_idx
from gpu.host import DeviceContext
from os import Atomic
from math import ceildiv
from sys.info import simd_width_of
from time.time import perf_counter


from extramojo.io.buffered import BufferedReader
from extramojo.bstr.bstr import SplitIterator


def span_to_int(span: Span[UInt8]) -> Int32:
    var acc: Int32 = 0
    for b in span:
        # print(acc, b)
        acc = (acc * 10) + (Int32(b) - 48)
    return acc


def cpu_method(side_a: Span[Int32], side_b: Span[Int32]) -> Int32:
    var sum: Int32 = 0

    @parameter
    fn sum_diffs[width: Int](i: Int):
        var a = side_a.unsafe_ptr().offset(i).load[width=width]()
        var b = side_b.unsafe_ptr().offset(i).load[width=width]()
        sum += abs(a - b).reduce_add()

    vectorize[sum_diffs, simd_width_of[Int32]()](len(side_a))
    return sum


def gpu_method(side_a: Span[Int32], side_b: Span[Int32]) -> Int32:
    ctx = DeviceContext()
    side_a_host = ctx.enqueue_create_host_buffer[Int32.dtype](len(side_a))
    side_b_host = ctx.enqueue_create_host_buffer[Int32.dtype](len(side_b))
    sum_host = ctx.enqueue_create_host_buffer[Int32.dtype](1)
    ctx.synchronize()
    side_a_dev = ctx.enqueue_create_buffer[Int32.dtype](len(side_a))
    side_b_dev = ctx.enqueue_create_buffer[Int32.dtype](len(side_b))
    sum_dev = ctx.enqueue_create_buffer[Int32.dtype](1)

    @parameter
    fn copy_mem[width: Int](i: Int):
        side_a_host.unsafe_ptr().offset(i).store(
            side_a.unsafe_ptr().offset(i).load[width=width]()
        )
        side_b_host.unsafe_ptr().offset(i).store(
            side_b.unsafe_ptr().offset(i).load[width=width]()
        )

    vectorize[copy_mem, simd_width_of[Int32]()](len(side_a))

    ctx.enqueue_copy(dst_buf=side_a_dev, src_buf=side_a_host)
    ctx.enqueue_copy(dst_buf=side_b_dev, src_buf=side_b_host)
    ctx.synchronize()

    alias block_size = 32
    alias coarse_factor = 32
    var num_blocks = ceildiv(len(side_a), block_size)
    ctx.enqueue_function[compute_diffs[block_size, coarse_factor],](
        side_a_dev.unsafe_ptr(),
        side_b_dev.unsafe_ptr(),
        UInt(len(side_a)),
        sum_dev.unsafe_ptr(),
        grid_dim=ceildiv(len(side_a), (coarse_factor * block_size)),
        block_dim=block_size,
    )

    ctx.enqueue_copy(dst_buf=sum_host, src_buf=sum_dev)
    ctx.synchronize()
    return sum_host[0]


fn compute_diffs[
    block_size: UInt, coarse_factor: UInt
](
    side_a: UnsafePointer[Scalar[DType.int32]],
    side_b: UnsafePointer[Scalar[DType.int32]],
    length: UInt,
    running_sum: UnsafePointer[Scalar[DType.int32]],
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x

    var thread_sum: Int32 = 0

    @parameter
    for i in range(UInt(0), coarse_factor):
        var idx = (global_i * coarse_factor) + i
        if idx < length:
            thread_sum += abs(side_a[idx] - side_b[idx])

    alias offsets = InlineArray[UInt32, size=5](16, 8, 4, 2, 1)

    @parameter
    for offset in range(0, len(offsets)):
        thread_sum += warp.shuffle_down(thread_sum, offsets[offset])

    if thread_idx.x == 0:
        _ = Atomic.fetch_add(running_sum, thread_sum)


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

    # part_a(side_a.copy(), side_b.copy())
    part_b(side_a.copy(), side_b.copy())


def part_a(var side_a: List[Int32], var side_b: List[Int32]):
    sort(side_a)
    sort(side_b)

    var cpu_start = perf_counter()
    var cpu_ans = cpu_method(side_a, side_b)
    var cpu_end = perf_counter()
    print("CPU Answer(", cpu_end - cpu_start, "s): ", cpu_ans)

    var gpu_start = perf_counter()
    var gpu_ans = gpu_method(side_a, side_b)
    var gpu_end = perf_counter()
    print("GPU Answer(", gpu_end - gpu_start, "s): ", gpu_ans)


def part_b(var side_a: List[Int32], var side_b: List[Int32]):
    # Need a count of the number of times a value in A appears in B
    # Answer is sum of times A value is in B for all values in A

    var cpu_start = perf_counter()
    var cpu_ans = part_b_cpu(side_a.copy(), side_b.copy())
    var cpu_end = perf_counter()
    print("CPU Answer(", cpu_end - cpu_start, "s): ", cpu_ans)

    # var gpu_start = perf_counter()
    # var gpu_ans = gpu_method(side_a, side_b)
    # var gpu_end = perf_counter()
    # print("GPU Answer(", gpu_end - gpu_start, "s): ", gpu_ans)


def part_b_cpu(var side_a: List[Int32], var side_b: List[Int32]) -> UInt:
    var b_counts: Dict[Int32, UInt] = {}
    for b in side_b:
        if b not in b_counts:
            b_counts[b] = 1
        else:
            b_counts[b] = b_counts[b] + 1

    var sum: UInt = 0
    for a in side_a:
        var count = b_counts.get(a, 0)
        sum += UInt(a) * count
    return sum
