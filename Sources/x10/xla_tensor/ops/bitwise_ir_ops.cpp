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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/bitwise_ir_ops.h"

#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/ops.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

template <typename F>
xla::XlaOp LowerBitwise(xla::XlaOp lhs, xla::XlaOp rhs, const F& bit_op) {
  std::tie(lhs, rhs) = XlaHelpers::PromoteValues(lhs, rhs);
  return bit_op(lhs, rhs);
}

}  // namespace

Value BitwiseAnd(const Value& node1, const Value& node2) {
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp result = LowerBitwise(
        op0, op1, [](xla::XlaOp lhs, xla::XlaOp rhs) { return lhs & rhs; });
    return node.ReturnOp(result, loctx);
  };
  auto shape_fn = [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return LowerBitwise(
        operands[0], operands[1],
        [](xla::XlaOp lhs, xla::XlaOp rhs) { return lhs & rhs; });
  };
  return GenericOp(
      OpKind(at::aten::__and__), {node1, node2},
      [&]() {
        return InferOutputShape({node1.shape(), node2.shape()}, shape_fn);
      },
      std::move(lower_fn));
}

Value BitwiseOr(const Value& node1, const Value& node2) {
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp result = LowerBitwise(
        op0, op1, [](xla::XlaOp lhs, xla::XlaOp rhs) { return lhs | rhs; });
    return node.ReturnOp(result, loctx);
  };
  auto shape_fn = [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return LowerBitwise(
        operands[0], operands[1],
        [](xla::XlaOp lhs, xla::XlaOp rhs) { return lhs | rhs; });
  };
  return GenericOp(
      OpKind(at::aten::__or__), {node1, node2},
      [&]() {
        return InferOutputShape({node1.shape(), node2.shape()}, shape_fn);
      },
      std::move(lower_fn));
}

Value BitwiseXor(const Value& node1, const Value& node2) {
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp result = LowerBitwise(
        op0, op1, [](xla::XlaOp lhs, xla::XlaOp rhs) { return lhs ^ rhs; });
    return node.ReturnOp(result, loctx);
  };
  auto shape_fn = [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return LowerBitwise(
        operands[0], operands[1],
        [](xla::XlaOp lhs, xla::XlaOp rhs) { return lhs ^ rhs; });
  };
  return GenericOp(
      OpKind(at::aten::__xor__), {node1, node2},
      [&]() {
        return InferOutputShape({node1.shape(), node2.shape()}, shape_fn);
      },
      std::move(lower_fn));
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
