from algorithm.functional import parallelize
from gpu.host import DeviceContext
from gpu import block_idx, thread_idx, block_dim, warp, syncwarp
from math import ceildiv
from memory import memcpy
from os import Atomic

from aoclib.parse import span_to_int
from extramojo.bstr.bstr import SplitIterator


@register_passable
@fieldwise_init
struct StartStop(ImplicitlyCopyable, Movable):
    var start: UInt64  # Inclusive
    var stop: UInt64  # Inclusive


def part_a():
    var fh = open("./big.txt", "r")
    var bytes = fh.read_bytes()

    var ranges = List[StartStop]()
    var sums = List[UInt64]()

    for rng in SplitIterator(bytes, ord(",")):
        var iter = SplitIterator(rng, ord("-"))
        var start = iter.__next__()
        var stop = iter.__next__()
        var start_num = span_to_int[DType.uint64](start)
        var stop_num = span_to_int[DType.uint64](stop)
        ranges.append(StartStop(start_num, stop_num))

    sums.resize(len(ranges), 0)

    fn count_invalids(idx: Int) capturing:
        var sum: UInt64 = 0
        var buffer = String()
        var rng = ranges[idx]
        for x in range(rng.start, rng.stop + 1):
            buffer.resize(0)
            x.write_to(buffer)
            if len(buffer) % 2 != 0:
                continue
            var midpoint = len(buffer) // 2

            if buffer[0:midpoint] == buffer[midpoint:]:
                sum += x
        sums[idx] = sum

    parallelize[count_invalids](len(ranges))
    var total: UInt64 = 0
    for s in sums:
        total += s

    print(total)


def part_b():
    var fh = open("./big.txt", "r")
    var bytes = fh.read_bytes()

    var ranges = List[StartStop]()
    var sums = List[UInt64]()

    for rng in SplitIterator(bytes, ord(",")):
        var iter = SplitIterator(rng, ord("-"))
        var start = iter.__next__()
        var stop = iter.__next__()
        var start_num = span_to_int[DType.uint64](start)
        var stop_num = span_to_int[DType.uint64](stop)

        ranges.append(StartStop(start_num, stop_num))

    sums.resize(len(ranges), 0)

    fn find_repeats(idx: Int) capturing:
        var rng = ranges[idx]
        var sum: UInt64 = 0
        var buffer = String()
        for x in range(rng.start, rng.stop + 1):
            buffer.resize(0)
            x.write_to(buffer)

            # Sliding window over the buffer, varying the size
            for i in range(len(buffer) - 1, 0, -1):
                var seq = buffer[0:i]
                for s in range(0, len(buffer), len(seq)):
                    if seq != buffer[s : s + len(seq)]:
                        break
                else:
                    sum += x
                    break
        sums[idx] = sum

    parallelize[find_repeats](len(ranges))

    var total: UInt64 = 0
    for s in sums:
        total += s

    print(total)


fn print_threads():
    print("Block index: [", block_idx.x, "]\tThread idx: [", thread_idx.x, "]")


fn find_repeats_gpu[
    block_size: UInt, coarse_factor: UInt
](
    starts: UnsafePointer[UInt64, MutAnyOrigin],
    stops: UnsafePointer[UInt64, MutAnyOrigin],
    length: Int,
    sum: UnsafePointer[UInt64, MutAnyOrigin],
):
    # Each warp will get 1 range, which should be between 0-32 numbers
    # each thread in the warp will work on 1 of the numbers in the range
    # shift down the sum, accumlate into one global locked sum

    # var global_i = block_dim.x * block_idx.x + thread_idx.x
    # Each block handles one warp; use block_idx.x as the warp/block ID
    var warp_id = block_idx.x

    var thread_sum: UInt64 = 0
    var buffer = String()

    @parameter
    for i in range(UInt(0), coarse_factor):
        var range_idx = (warp_id * coarse_factor) + i
        if range_idx >= UInt(length):
            continue

        var start = starts[range_idx]
        var stop = stops[range_idx]
        # now get the num for each of th warp threads
        var num = start + thread_idx.x
        # print(start, "-", stop, ":", num)

        if num <= stop:
            # Do the thing, start = idx * 2, stop = (idx * 2) + 1
            buffer.resize(0)
            num.write_to(buffer)

            # Sliding window over the buffer, varying the size
            for seq_len in range(len(buffer) - 1, 0, -1):
                var seq = buffer[0:seq_len]
                for pos in range(0, len(buffer), len(seq)):
                    if seq != buffer[pos : pos + len(seq)]:
                        break
                else:
                    thread_sum += num
                    break

    # Skip shuffle reduction for now - just atomic add each thread's sum directly
    if thread_sum > 0:
        _ = Atomic.fetch_add(sum, thread_sum)


def part_b_gpu():
    var fh = open("./big.txt", "r")
    var bytes = fh.read_bytes()

    var ctx = DeviceContext()

    var starts = List[UInt64]()
    var stops = List[UInt64]()

    for rng in SplitIterator(bytes, ord(",")):
        var iter = SplitIterator(rng, ord("-"))
        var start = iter.__next__()
        var stop = iter.__next__()
        var start_num = span_to_int[DType.uint64](start)
        var stop_num = span_to_int[DType.uint64](stop)

        # split the ranges into lengths of 32 ish (1 warp size)
        # Use stop_num + 1 to ensure we create sub-ranges covering up to stop_num inclusive
        for i in range(start_num, stop_num + 1, 32):
            starts.append(i)
            stops.append(min(stop_num, i + 31))

    var starts_host = ctx.enqueue_create_host_buffer[DType.uint64](len(starts))
    var stops_host = ctx.enqueue_create_host_buffer[DType.uint64](len(stops))
    var sum_host = ctx.enqueue_create_host_buffer[DType.uint64](1)
    ctx.synchronize()

    var starts_dev = ctx.enqueue_create_buffer[DType.uint64](len(starts))
    var stops_dev = ctx.enqueue_create_buffer[DType.uint64](len(stops))
    var sum_dev = ctx.enqueue_create_buffer[DType.uint64](1)
    memcpy(
        dest=starts_host.unsafe_ptr(),
        src=starts.unsafe_ptr(),
        count=len(starts),
    )
    memcpy(
        dest=stops_host.unsafe_ptr(), src=stops.unsafe_ptr(), count=len(stops)
    )

    starts_host.enqueue_copy_to(starts_dev)
    stops_host.enqueue_copy_to(stops_dev)
    ctx.synchronize()

    # Still need a sum location for host to copy to
    alias block_size = 32
    alias coarse_size = 32
    var kernel = ctx.compile_function_checked[
        find_repeats_gpu[block_size, coarse_size],
        find_repeats_gpu[block_size, coarse_size],
    ]()
    ctx.enqueue_function_checked(
        kernel,
        starts_dev,
        stops_dev,
        len(starts),
        sum_dev,
        grid_dim=ceildiv(
            len(starts), coarse_size
        ),  # Each block handles coarse_size sub-ranges
        block_dim=block_size,  # 1 warp
    )
    ctx.synchronize()
    sum_dev.enqueue_copy_to(sum_host)
    ctx.synchronize()
    print(sum_host[0])


def main():
    # part_a()
    # print("gpu")
    # part_b_gpu()
    # print("cpu")
    part_b()
