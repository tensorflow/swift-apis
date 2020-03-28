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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/xla_pad.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& operand, const Value& padding_value,
                           const xla::PaddingConfig& padding) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    CHECK_EQ(operands.size(), 2)
        << "Unexpected number of operands: " << operands.size();
    return xla::Pad(operands[0], operands[1], padding);
  };
  return InferOutputShape({operand.shape(), padding_value.shape()},
                          lower_for_shape_fn);
}

xla::hash_t PaddingConfigHash(const xla::PaddingConfig& padding_config) {
  std::vector<xla::int64> low;
  std::vector<xla::int64> high;
  std::vector<xla::int64> interior;
  for (const xla::PaddingConfig::PaddingConfigDimension& dim_padding :
       padding_config.dimensions()) {
    low.push_back(dim_padding.edge_padding_low());
    high.push_back(dim_padding.edge_padding_high());
    interior.push_back(dim_padding.interior_padding());
  }
  return xla::util::MHash(low, high, interior);
}

}  // namespace

XlaPad::XlaPad(const Value& operand, const Value& padding_value,
               xla::PaddingConfig padding_config)
    : Node(ir::OpKind(at::aten::xla_pad), {operand, padding_value},
           [&]() {
             return NodeOutputShape(operand, padding_value, padding_config);
           },
           /*num_outputs=*/1, PaddingConfigHash(padding_config)),
      padding_config_(std::move(padding_config)) {}

NodePtr XlaPad::Clone(OpList operands) const {
  return MakeNode<XlaPad>(operands.at(0), operands.at(1), padding_config_);
}

XlaOpVector XlaPad::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp padding_value = loctx->GetOutputOp(operand(1));
  xla::XlaOp output = xla::Pad(input, padding_value, padding_config_);
  return ReturnOp(output, loctx);
}

std::string XlaPad::ToString() const {
  std::stringstream ss;
  ss << Node::ToString()
     << ", padding=" << xla::PaddingConfigToString(padding_config_);
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
