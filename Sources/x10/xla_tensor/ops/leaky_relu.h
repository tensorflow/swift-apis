#pragma once

#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"

namespace swift_xla {
namespace ir {
namespace ops {

class LeakyRelu : public Node {
 public:
  LeakyRelu(const Value& input, double negative_slope);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  double negative_slope() const { return negative_slope_; }

 private:
  double negative_slope_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
