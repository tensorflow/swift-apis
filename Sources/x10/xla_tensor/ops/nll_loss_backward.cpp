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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/nll_loss_backward.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/nll_loss.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& grad_output, const Value& logits,
                           const Value& labels,
                           const absl::optional<Value>& weight,
                           const absl::optional<Value>& total_weight,
                           ReductionMode reduction, int ignore_index) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    absl::optional<xla::XlaOp> weight;
    absl::optional<xla::XlaOp> total_weight;
    if (operands.size() > 3) {
      XLA_CHECK_EQ(operands.size(), 5)
          << "If weight is specified, so must be total_weight";
      weight = operands[3];
      total_weight = operands[4];
    }
    return BuildNllLossBackward(operands[0], operands[1], operands[2], weight,
                                total_weight, ignore_index, reduction);
  };
  std::vector<xla::Shape> shapes;
  for (auto& input : xla::util::GetValuesVector<Value>(
           {grad_output, logits, labels}, {&weight, &total_weight})) {
    shapes.push_back(input.shape());
  }
  return InferOutputShape(shapes, lower_for_shape_fn);
}

}  // namespace

NllLossBackward::NllLossBackward(const Value& grad_output, const Value& logits,
                                 const Value& labels,
                                 const absl::optional<Value>& weight,
                                 const absl::optional<Value>& total_weight,
                                 ReductionMode reduction, int ignore_index)
    : Node(ir::OpKind(at::aten::nll_loss_backward),
           xla::util::GetValuesVector<Value>({grad_output, logits, labels},
                                             {&weight, &total_weight}),
           [&]() {
             return NodeOutputShape(grad_output, logits, labels, weight,
                                    total_weight, reduction, ignore_index);
           },
           /*num_outputs=*/1,
           xla::util::MHash(xla::util::GetEnumValue(reduction), ignore_index)),
      reduction_(reduction),
      ignore_index_(ignore_index) {}

NodePtr NllLossBackward::Clone(OpList operands) const {
  absl::optional<Value> weight;
  absl::optional<Value> total_weight;
  if (operands.size() > 3) {
    weight = operands.at(3);
    total_weight = operands.at(4);
  }
  return MakeNode<NllLossBackward>(operands.at(0), operands.at(1),
                                   operands.at(2), weight, total_weight,
                                   reduction_, ignore_index_);
}

XlaOpVector NllLossBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp logits = loctx->GetOutputOp(operand(1));
  xla::XlaOp labels = loctx->GetOutputOp(operand(2));
  absl::optional<xla::XlaOp> weight;
  absl::optional<xla::XlaOp> total_weight;
  if (operands().size() > 3) {
    weight = loctx->GetOutputOp(operand(3));
    total_weight = loctx->GetOutputOp(operand(4));
  }
  return ReturnOp(BuildNllLossBackward(grad_output, logits, labels, weight,
                                       total_weight, ignore_index_, reduction_),
                  loctx);
}

std::string NllLossBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString()
     << ", reduction=" << xla::util::GetEnumValue(reduction_)
     << ", ignore_index=" << ignore_index_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
