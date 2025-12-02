
@always_inline
fn span_to_int[dtype: DType = DType.int32](span: Span[UInt8]) -> Scalar[dtype]:
    var acc: Scalar[dtype] = 0
    for b in span:
        # print(acc, b)
        acc = (acc * 10) + (Scalar[dtype](b) - 48)
    return acc