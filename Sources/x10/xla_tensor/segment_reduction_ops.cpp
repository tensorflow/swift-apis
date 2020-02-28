#include "tensorflow/compiler/tf2xla/xla_tensor/segment_reduction_ops.h"

#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/lib/scatter.h"

namespace swift_xla {

xla::XlaOp UnsortedSegmentReduce(
    xla::XlaOp data, xla::XlaOp indices, xla::XlaOp init_value,
    xla::int64 num_segments,
    const std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp)>& combine) {
  const xla::Shape& data_shape = XlaHelpers::ShapeOfXlaOp(data);
  const xla::Shape& indices_shape = XlaHelpers::ShapeOfXlaOp(indices);
  const auto data_size = data_shape.dimensions();
  std::vector<xla::int64> buffer_size(data_size.begin() + indices_shape.rank(),
                                      data_size.end());
  buffer_size.insert(buffer_size.begin(), num_segments);
  xla::XlaOp buffer = xla::Broadcast(init_value, buffer_size);
  auto combiner = [&combine](xla::XlaOp a, xla::XlaOp b,
                             xla::XlaBuilder* builder) {
    return combine(a, b);
  };
  return ConsumeValue(tensorflow::XlaScatter(buffer, /*updates=*/data, indices,
                                             /*indices_are_vectors=*/false,
                                             combiner, data.builder()));
}

}  // namespace swift_xla
