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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/arithmetic_ir_ops.h"

#include <memory>

#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/ops.h"

namespace swift_xla {
namespace ir {

NodePtr operator+(const Value& node1, const Value& node2) {
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(XlaHelpers::PromotedAdd(op0, op1), loctx);
  };
  return ops::GenericOp(
      OpKind(at::aten::add), {node1, node2},
      XlaHelpers::GetPromotedBinaryOpShape(node1.shape(), node2.shape()),
      std::move(lower_fn));
}

NodePtr operator-(const Value& node1, const Value& node2) {
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(XlaHelpers::PromotedSub(op0, op1), loctx);
  };
  return ops::GenericOp(
      OpKind(at::aten::sub), {node1, node2},
      XlaHelpers::GetPromotedBinaryOpShape(node1.shape(), node2.shape()),
      std::move(lower_fn));
}

NodePtr operator*(const Value& node1, const Value& node2) {
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(XlaHelpers::PromotedMul(op0, op1), loctx);
  };
  return ops::GenericOp(
      OpKind(at::aten::mul), {node1, node2},
      XlaHelpers::GetPromotedBinaryOpShape(node1.shape(), node2.shape()),
      std::move(lower_fn));
}

NodePtr operator/(const Value& node1, const Value& node2) {
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(XlaHelpers::PromotedDiv(op0, op1), loctx);
  };
  return ops::GenericOp(
      OpKind(at::aten::div), {node1, node2},
      XlaHelpers::GetPromotedBinaryOpShape(node1.shape(), node2.shape()),
      std::move(lower_fn));
}

}  // namespace ir
}  // namespace swift_xla
