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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/as_strided.h"

#include <algorithm>

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/data_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/tensor_util.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::XlaOp LowerAsStrided(xla::XlaOp input, absl::Span<const xla::int64> size,
                          absl::Span<const xla::int64> stride,
                          xla::int64 storage_offset) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::int64 input_element_count = xla::ShapeUtil::ElementsIn(input_shape);
  xla::int64 slice_size = xla::util::Multiply<xla::int64>(size);
  XLA_CHECK_LE(storage_offset + slice_size, input_element_count);

  xla::XlaOp off_input = input;
  if (storage_offset > 0 || slice_size < input_element_count) {
    xla::XlaOp r1_input = XlaHelpers::Flatten(input);
    off_input = xla::SliceInDim(r1_input, storage_offset,
                                storage_offset + slice_size, 1, 0);
  }

  std::vector<xla::int64> permutation = xla::InversePermutation(
      AsStrided::GetArrayStridePermutation(stride, size));
  std::vector<xla::int64> new_sizes = xla::Permute(permutation, size);
  xla::XlaOp reshaped_input = XlaHelpers::DynamicReshape(off_input, new_sizes);
  return xla::IsIdentityPermutation(permutation)
             ? reshaped_input
             : xla::Transpose(reshaped_input, permutation);
}

}  // namespace

AsStrided::AsStrided(const Value& input, std::vector<xla::int64> size,
                     std::vector<xla::int64> stride, xla::int64 storage_offset)
    : Node(ir::OpKind(at::aten::as_strided), {input},
           [&]() {
             return xla::ShapeUtil::MakeShape(input.shape().element_type(),
                                              size);
           },
           /*num_outputs=*/1, xla::util::MHash(size, stride, storage_offset)),
      size_(std::move(size)),
      stride_(std::move(stride)),
      storage_offset_(storage_offset) {}

std::string AsStrided::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", size=(" << absl::StrJoin(size_, ", ")
     << "), stride=(" << absl::StrJoin(stride_, ", ")
     << "), storage_offset=" << storage_offset_;
  return ss.str();
}

NodePtr AsStrided::Clone(OpList operands) const {
  return MakeNode<AsStrided>(operands.at(0), size_, stride_, storage_offset_);
}

XlaOpVector AsStrided::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(LowerAsStrided(input, size_, stride_, storage_offset_),
                  loctx);
}

bool AsStrided::StrideIsSupported(const xla::Shape& input_shape,
                                  absl::Span<const xla::int64> size,
                                  absl::Span<const xla::int64> stride,
                                  xla::int64 storage_offset) {
  std::vector<xla::int64> sorted_stride(stride.begin(), stride.end());
  std::sort(sorted_stride.begin(), sorted_stride.end());
  return stride.empty() || sorted_stride.front() == 1;
}

std::vector<xla::int64> AsStrided::GetArrayStridePermutation(
    absl::Span<const xla::int64> stride, absl::Span<const xla::int64> size) {
  std::vector<xla::int64> permutation =
      xla::util::Iota<xla::int64>(stride.size());
  std::sort(permutation.begin(), permutation.end(),
            [&](xla::int64 a, xla::int64 b) { return stride[a] > stride[b]; });
  return permutation;
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
