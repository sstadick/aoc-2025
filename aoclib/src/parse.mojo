
fn span_to_int(span: Span[UInt8]) -> Int32:
    var acc: Int32 = 0
    for b in span:
        # print(acc, b)
        acc = (acc * 10) + (Int32(b) - 48)
    return acc