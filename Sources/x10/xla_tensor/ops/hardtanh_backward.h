#pragma once

#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"

namespace swift_xla {
namespace ir {
namespace ops {

class HardtanhBackward : public Node {
 public:
  HardtanhBackward(const Value& grad_output, const Value& input,
                   at::Scalar min_val, at::Scalar max_val);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  at::Scalar min_val() const { return min_val_; }

  at::Scalar max_val() const { return max_val_; }

 private:
  at::Scalar min_val_;
  at::Scalar max_val_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
