from algorithm.functional import vectorize
from gpu import syncwarp, warp
from gpu.id import block_dim, block_idx, thread_idx
from gpu.host import DeviceContext, ConstantMemoryMapping
from os import Atomic
from math import ceildiv
from sys.info import simd_width_of
from time.time import perf_counter
from memory import memcpy, stack_allocation
from layout import Layout, LayoutTensor


from aoclib.parse import span_to_int
from extramojo.io.buffered import BufferedReader
from extramojo.bstr.bstr import SplitIterator


# Example using constant mem: https://github.com/modular/modular/blob/b8756654dd050be664396757be2fc7c495484e1b/max/kernels/test/gpu/basics/test_constant_memory.mojo#L105

# alias file = "./big.txt"
# alias rows = 135
# alias cols = 135
alias file = "./big.txt"
alias rows = 135
alias cols = 135
alias layout = Layout.row_major(rows, cols)


fn make_mask[
    r: Int
]() -> UnsafePointer[
    Byte, ImmutAnyOrigin, address_space = AddressSpace.CONSTANT
]:
    """Create a mask that will mulitply by 1 for all things around the center.

    1 1 1
    1 0 1
    1 1 1
    """

    alias width = (2 * r) + 1
    alias height = width
    var ptr = stack_allocation[
        width * height, Byte, address_space = AddressSpace.CONSTANT
    ]()

    @parameter
    for i in range(0, width * height):
        ptr[i] = 1

    ptr[(width * height) // 2] = 0
    return ptr


fn roll_count_kernel(
    output: UnsafePointer[Byte, MutAnyOrigin], length: UInt, width: UInt
):
    alias mask_r = 1
    alias mask = make_mask[mask_r]()
    alias mask_width = (2 * mask_r) + 1

    var input = stack_allocation[
        rows * cols,
        Byte,
        address_space = AddressSpace.CONSTANT,
        name = StaticString("input"),
    ]()

    var out_col = block_idx.x * block_dim.x + thread_idx.x
    var out_row = block_idx.y * block_dim.y + thread_idx.y

    if out_row >= length or out_col >= width:
        return

    var base_value = input[out_row * width + out_col]

    var out_value: UInt8 = (
        base_value  # starts with 1 for rolls, 0 for non-rolls
    )
    for f_row in range(UInt(0), UInt(3)):
        for f_col in range(UInt(0), UInt(3)):
            var in_row = out_row - 1 + f_row
            var in_col = out_col - 1 + f_col

            if (
                in_row >= 0
                and in_row < length
                and in_col >= 0
                and in_col < width
            ):
                # In here test if roll or not via base_value (1 if roll, 0 if not)
                out_value += (
                    mask[f_row * mask_width + f_col]
                    * base_value
                    * input[in_row * width + in_col]
                )

    output[out_row * width + out_col] = out_value


def read_data() -> LayoutTensor[DType.uint8, layout, MutOrigin.external]:
    var fh = open(file, "r")  # 10x10

    var ptr = alloc[Byte](rows * cols)

    var reader = BufferedReader(fh^)
    var buffer = List[UInt8](capacity=1024)

    var row_num = 0
    while reader.read_until(buffer, char=UInt(ord("\n"))):

        @parameter
        fn zero_base[width: Int](i: Int):
            alias roll = SIMD[DType.uint8, width](ord("@"))
            var items = buffer.unsafe_ptr().offset(i).load[width=width]()
            var rolls = items.eq(roll).cast[DType.uint8]()
            ptr.offset((row_num * cols) + i).store(rolls)

        vectorize[zero_base, simd_width_of[DType.uint8]()](len(buffer))
        row_num += 1
        buffer.clear()

    var layout_tensor = LayoutTensor[DType.uint8, layout](ptr)

    return layout_tensor


def part_a():
    var data = read_data()

    var ctx = DeviceContext()

    # Every "roll" will start off with a count of one
    var output_host = ctx.enqueue_create_host_buffer[DType.uint8](rows * cols)
    memcpy(dest=output_host.unsafe_ptr(), src=data.ptr, count=rows * cols)
    var output_dev = ctx.enqueue_create_buffer[DType.uint8](rows * cols)
    output_dev.enqueue_copy_from(output_host)

    alias kernel = roll_count_kernel
    ctx.enqueue_function_checked[kernel, kernel](
        output_dev,
        UInt(rows),
        UInt(cols),
        grid_dim=(ceildiv(cols, 16), ceildiv(rows, 16)),
        block_dim=(16, 16),
        constant_memory=[
            ConstantMemoryMapping(
                "input", data.ptr.bitcast[NoneType](), rows * cols
            )
        ],
    )
    output_host.enqueue_copy_from(output_dev)
    ctx.synchronize()

    alias simd_width = simd_width_of[DType.uint8]()
    var final: UInt64 = 0

    @parameter
    fn sum_up[width: Int](i: Int):
        alias five = SIMD[DType.uint8, width](
            5
        )  # offset to account for marked outputs that _are_ rolls
        alias one = SIMD[DType.uint8, width](1)
        var output_val = output_host.unsafe_ptr().offset(i).load[width=width]()
        var temp = (output_val.ge(one) & output_val.lt(five)).cast[
            DType.uint8
        ]()

        # have to widen bc overflows
        @parameter
        for i in range(0, width):
            final += UInt64(temp[i])

    vectorize[sum_up, simd_width](rows * cols)

    print("Answer:", final)


def part_b():
    var data = read_data()

    var ctx = DeviceContext()
    var running_count: UInt64 = 0

    var output_host = ctx.enqueue_create_host_buffer[DType.uint8](rows * cols)
    var output_dev = ctx.enqueue_create_buffer[DType.uint8](rows * cols)

    # This is a cop out, mutation should happen on the GPU but I put the data in global
    # mem to avoid tiling. Maybe I didn't need to avoid that in the first place?
    while True:
        # Every "roll" will start off with a count of one
        memcpy(dest=output_host.unsafe_ptr(), src=data.ptr, count=rows * cols)
        output_dev.enqueue_copy_from(output_host)

        alias kernel = roll_count_kernel
        ctx.enqueue_function_checked[kernel, kernel](
            output_dev,
            UInt(rows),
            UInt(cols),
            grid_dim=(ceildiv(cols, 16), ceildiv(rows, 16)),
            block_dim=(16, 16),
            constant_memory=[
                ConstantMemoryMapping(
                    "input", data.ptr.bitcast[NoneType](), rows * cols
                )
            ],
        )
        output_host.enqueue_copy_from(output_dev)
        ctx.synchronize()

        alias simd_width = simd_width_of[DType.uint8]()
        var final: UInt64 = 0

        @parameter
        fn sum_up[width: Int](i: Int):
            alias five = SIMD[DType.uint8, width](
                5
            )  # offset to account for marked outputs that _are_ rolls
            alias one = SIMD[DType.uint8, width](1)
            var output_val = (
                output_host.unsafe_ptr().offset(i).load[width=width]()
            )
            var temp = (output_val.ge(one) & output_val.lt(five)).cast[
                DType.uint8
            ]()

            var original = data.ptr.offset(i).load[width=width]()
            data.ptr.offset(i).store(original ^ temp)

            # have to widen bc overflows
            @parameter
            for i in range(0, width):
                final += UInt64(temp[i])

        vectorize[sum_up, simd_width](rows * cols)

        if final == 0:
            break
        running_count += final

    print("Answer:", running_count)


def main():
    part_a()
