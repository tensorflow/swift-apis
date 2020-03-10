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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/einsum.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/data_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(absl::Span<const ir::Value> values,
                           const std::string& equation) {
  XLA_CHECK_EQ(values.size(), 2)
      << "Only two inputs supported for einsum for now";
  auto lower_for_shape_fn =
      [equation](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return xla::Einsum(operands[0], operands[1], equation,
                       XlaHelpers::mat_mul_precision());
  };
  std::vector<xla::Shape> shapes;
  shapes.reserve(values.size());
  for (auto& value : values) {
    shapes.push_back(value.shape());
  }
  return InferOutputShape(shapes, lower_for_shape_fn);
}

}  // namespace

Einsum::Einsum(const std::string& equation, absl::Span<const ir::Value> values)
    : Node(ir::OpKind(at::aten::einsum), values,
           [&]() { return NodeOutputShape(values, equation); },
           /*num_outputs=*/1, xla::util::MHash(equation)),
      equation_(equation) {}

NodePtr Einsum::Clone(OpList operands) const {
  return MakeNode<Einsum>(equation_, operands);
}

XlaOpVector Einsum::Lower(LoweringContext* loctx) const {
  xla::XlaOp output = xla::Einsum(loctx->GetOutputOp(operand(0)),
                                  loctx->GetOutputOp(operand(1)), equation_,
                                  XlaHelpers::mat_mul_precision());
  return ReturnOp(output, loctx);
}

std::string Einsum::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", equation=" << equation_;
  return ss.str();
}

bool Einsum::SupportsEquation(const std::string& equation, xla::int64 x_rank,
                              xla::int64 y_rank) {
  auto einsum_config_numeric_or_err =
      xla::ParseEinsumString(equation, x_rank, y_rank);
  if (!einsum_config_numeric_or_err.ok()) {
    return false;
  }
  auto einsum_config_numeric = einsum_config_numeric_or_err.ConsumeValueOrDie();
  auto validation_status = xla::ValidateEinsumNumericDimensions(
      /*x_config=*/einsum_config_numeric[0],
      /*y_config=*/einsum_config_numeric[1],
      /*output_config=*/einsum_config_numeric[2]);
  return validation_status.ok();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
