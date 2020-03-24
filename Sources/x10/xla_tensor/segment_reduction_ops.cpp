// Copyright 2020 TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
