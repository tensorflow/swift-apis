#ifndef X10_XLA_TENSOR_SEGMENT_REDUCTION_OPS_H_
#define X10_XLA_TENSOR_SEGMENT_REDUCTION_OPS_H_

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace swift_xla {

xla::XlaOp UnsortedSegmentReduce(
    xla::XlaOp data, xla::XlaOp indices, xla::XlaOp init_value,
    xla::int64 num_segments,
    const std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp)>& combine);

}  // namespace swift_xla

#endif  // X10_XLA_TENSOR_SEGMENT_REDUCTION_OPS_H_
