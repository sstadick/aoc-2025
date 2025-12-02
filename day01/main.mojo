from time import perf_counter

from extramojo.io.buffered import BufferedReader
from extramojo.bstr.bstr import SplitIterator

alias RIGHT = Direction(False)
alias LEFT = Direction(True)

@always_inline
fn span_to_int(span: Span[UInt8]) -> Int32:
    var acc: Int32 = 0
    for b in span:
        # print(acc, b)
        acc = (acc * 10) + (Int32(b) - 48)
    return acc

@fieldwise_init
struct Direction(Copyable, ImplicitlyCopyable, Movable):
    var _points_left: Bool

    @always_inline
    fn is_left(self) -> Bool:
        return self._points_left

    @always_inline
    fn is_right(self) -> Bool:
        return not self.is_left()

    @staticmethod
    fn from_byte(byte: UInt8) -> Self:
        if byte == ord("L"):
            return Self(True)
        else:
            return Self(False)


@fieldwise_init
struct ComboDir:
    var storage: Int
    var dir: Direction

    @always_inline
    fn __init__(out self, dir: Direction, value: Int):
        self.storage = value
        self.dir = dir

    @always_inline
    fn update_pos(self, pos: Int) -> Int:
        if self.dir.is_left():
            return (pos - self.storage) % 100
        else:
            return (pos + self.storage) % 100

    fn updater(self, pos: Int) -> UpdatedPos:
        var temp: Int

        var full_spins = self.storage // 100
        var remainder = self.storage - (100 * full_spins)

        if self.dir.is_left():
            temp = pos - remainder
        else:
            temp = pos + remainder

        count = 0

        if temp >= 100:
            temp = temp - 100
            count += 1
        elif temp == 0:
            count += 1
        elif temp < 0:
            temp = temp + 100
            if pos != 0:
                count += 1

        count += full_spins
        return UpdatedPos(temp, count)


@fieldwise_init
struct UpdatedPos(ImplicitlyCopyable, Movable):
    var new_pos: Int
    var times_zero: Int


def part_a() -> Int:
    var fh = open("./big.txt", "r")
    var reader = BufferedReader(fh^)
    var buffer = List[UInt8](capacity=16)

    var pos = 50  # The initial dial pos
    var count = 0
    while reader.read_until(buffer, char=UInt(ord("\n"))):
        var dir = Direction.from_byte(buffer[0])
        var num = span_to_int(buffer[1:])
        var combo = ComboDir(dir, Int(num))
        pos = combo.update_pos(pos)
        count += Int(pos == 0)
        buffer.clear()
    return count


def part_b() -> Int:
    var fh = open("./big.txt", "r")
    var reader = BufferedReader(fh^)
    var buffer = List[UInt8](capacity=16)

    var pos = 50  # The initial dial pos
    var count = 0
    while reader.read_until(buffer, char=UInt(ord("\n"))):
        var dir = Direction.from_byte(buffer[0])
        var num = span_to_int(buffer[1:])
        var combo = ComboDir(dir, Int(num))
        var updated = combo.updater(pos)
        pos = updated.new_pos
        count += updated.times_zero
        buffer.clear()
    return count


def main():
    var start = perf_counter()
    var count = part_b()
    var end = perf_counter()
    print(count, "in", end - start)
