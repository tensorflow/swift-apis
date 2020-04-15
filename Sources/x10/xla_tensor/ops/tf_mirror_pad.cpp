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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/tf_mirror_pad.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/data_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input,
                           absl::Span<const xla::int64> padding,
                           tensorflow::MirrorPadMode mode) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildMirrorPad(operands[0], padding, mode);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

TfMirrorPad::TfMirrorPad(const Value& input, std::vector<xla::int64> padding,
                         tensorflow::MirrorPadMode mode)
    : Node(OpKind(at::aten::tf_mirror_pad), {input},
           [&]() { return NodeOutputShape(input, padding, mode); },
           /*num_outputs=*/1,
           xla::util::MHash(padding, static_cast<int>(mode))),
      padding_(std::move(padding)),
      mode_(mode) {}

NodePtr TfMirrorPad::Clone(OpList operands) const {
  return MakeNode<TfMirrorPad>(operands.at(0), padding_, mode_);
}

XlaOpVector TfMirrorPad::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildMirrorPad(input, padding_, mode_);
  return ReturnOp(output, loctx);
}

std::string TfMirrorPad::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", padding=(" << absl::StrJoin(padding_, ", ")
     << "), mode=" << static_cast<int>(mode_);
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
