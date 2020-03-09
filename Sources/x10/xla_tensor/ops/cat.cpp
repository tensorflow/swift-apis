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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/cat.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/data_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(absl::Span<const ir::Value> values, xla::int64 dim) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildCat(operands, dim);
  };
  std::vector<xla::Shape> shapes;
  shapes.reserve(values.size());
  for (auto& value : values) {
    shapes.push_back(value.shape());
  }
  return InferOutputShape(shapes, lower_for_shape_fn);
}

}  // namespace

Cat::Cat(absl::Span<const ir::Value> values, xla::int64 dim)
    : Node(ir::OpKind(at::aten::cat), values,
           [&]() { return NodeOutputShape(values, dim); },
           /*num_outputs=*/1, xla::util::MHash(dim)),
      dim_(dim) {}

NodePtr Cat::Clone(OpList operands) const {
  return MakeNode<Cat>(operands, dim_);
}

XlaOpVector Cat::Lower(LoweringContext* loctx) const {
  std::vector<xla::XlaOp> inputs;
  for (auto& operand : operands()) {
    inputs.push_back(loctx->GetOutputOp(operand));
  }
  return ReturnOp(BuildCat(inputs, dim_), loctx);
}

std::string Cat::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
