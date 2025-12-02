from aoclib.parse import span_to_int
from extramojo.bstr.bstr import SplitIterator


def part_a():
    var fh = open("./big.txt", "r")
    var bytes = fh.read_bytes()

    var sum: UInt64 = 0
    var buffer = String()
    for rng in SplitIterator(bytes, ord(",")):
        var iter = SplitIterator(rng, ord("-"))
        var start = iter.__next__()
        var stop = iter.__next__()
        var start_num = span_to_int[DType.uint64](start)
        var stop_num = span_to_int[DType.uint64](stop)

        for x in range(start_num, stop_num+1):
            buffer.resize(0)
            x.write_to(buffer)
            if len(buffer) % 2 != 0:
                continue
            var midpoint = len(buffer) // 2

            if buffer[0:midpoint] == buffer[midpoint:]:
                sum += x
    print(sum)

def part_b():
    var fh = open("./big.txt", "r")
    var bytes = fh.read_bytes()

    var sum: UInt64 = 0
    var buffer = String()
    for rng in SplitIterator(bytes, ord(",")):
        var iter = SplitIterator(rng, ord("-"))
        var start = iter.__next__()
        var stop = iter.__next__()
        var start_num = span_to_int[DType.uint64](start)
        var stop_num = span_to_int[DType.uint64](stop)

        for x in range(start_num, stop_num+1):
            # print(x, ":", bin(x))
            buffer.resize(0)
            x.write_to(buffer)

            # Sliding window over the buffer, varying the size
            for i in range(1, len(buffer)):
                var seq = buffer[0:i]
                var equal = False
                for s in range(0, len(buffer), len(seq)):
                    if seq != buffer[s:s+len(seq)]:
                        break
                else:
                    equal = True
                
                if equal:
                    sum += x
                    break 

    print(sum)


def main():
    part_b()