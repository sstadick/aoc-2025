from algorithm.functional import parallelize

from aoclib.parse import span_to_int
from extramojo.bstr.bstr import SplitIterator

@register_passable
@fieldwise_init
struct StartStop(ImplicitlyCopyable, Movable):
    var start: UInt64 # Inclusive
    var stop: UInt64 # Inclusive

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
        for x in range(rng.start, rng.stop+1):
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
        for x in range(rng.start, rng.stop+1):
            buffer.resize(0)
            x.write_to(buffer)

            # Sliding window over the buffer, varying the size
            for i in range(len(buffer)-1, 0, -1):
                var seq = buffer[0:i]
                for s in range(0, len(buffer), len(seq)):
                    if seq != buffer[s:s+len(seq)]:
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


def main():
    part_a()
    # part_b()