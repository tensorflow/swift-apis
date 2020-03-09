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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/diagonal_view_update.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/matrix.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/xla_ops.h"

namespace swift_xla {
namespace ir {
namespace ops {

DiagonalViewUpdate::DiagonalViewUpdate(const Value& target, const Value& input,
                                       xla::int64 offset, xla::int64 dim1,
                                       xla::int64 dim2)
    : Node(xla_diagonal_view_update, {target, input}, target.shape(),
           /*num_outputs=*/1, xla::util::MHash(offset, dim1, dim2)),
      offset_(offset),
      dim1_(dim1),
      dim2_(dim2) {}

NodePtr DiagonalViewUpdate::Clone(OpList operands) const {
  return MakeNode<DiagonalViewUpdate>(operands.at(0), operands.at(1), offset_,
                                      dim1_, dim2_);
}

XlaOpVector DiagonalViewUpdate::Lower(LoweringContext* loctx) const {
  xla::XlaOp target = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  xla::XlaOp output =
      BuildDiagonalViewUpdate(target, input, offset_, dim1_, dim2_);
  return ReturnOp(output, loctx);
}

std::string DiagonalViewUpdate::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", offset=" << offset_ << ", dim1=" << dim1_
     << ", dim2=" << dim2_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
