from aoclib.parse import span_to_int
from extramojo.io.buffered import BufferedReader


def part_a():
    var fh = open("./big.txt", "r")
    var reader = BufferedReader(fh^)
    var buffer = List[UInt8](capacity=1024)

    var sum = 0

    while reader.read_until(buffer, char=UInt(ord("\n"))):
        var largest = 0
        var second_largest = 0
        for i, b in enumerate(buffer):
            var num = b - 48
            if num > largest and i != len(buffer) - 1:
                largest = Int(num)
                second_largest = 0
            elif num > second_largest:
                second_largest = Int(num)

        var total = (largest * 10) + second_largest
        # print(total)
        sum += total
        buffer.clear()
    print(sum)


def part_b():
    var fh = open("./big.txt", "r")
    var reader = BufferedReader(fh^)
    var buffer = List[UInt8](capacity=1024)

    var sum: UInt64 = 0

    while reader.read_until(buffer, char=UInt(ord("\n"))):
        var total = joltage[12](buffer)
        # print(total)
        sum += total
        buffer.clear()
    print(sum)


fn joltage[size: Int](values: Span[Byte]) -> UInt64:
    var jolts = InlineArray[Byte, size](fill=0)

    for i in range(0, len(values)):
        var num = values[i] - 48

        @parameter
        for j in range(0, size):
            if num > jolts[j] and len(values) - i >= size - j:
                jolts[j] = num

                # Set any previously set values to 0 that come after this
                @parameter
                for k in range(j + 1, size):
                    jolts[k] = 0
                break

    var total: UInt64 = 0

    @parameter
    for i in range(0, size):
        total = (total * 10) + UInt64(jolts[i])

    return total


def main():
    part_a()
