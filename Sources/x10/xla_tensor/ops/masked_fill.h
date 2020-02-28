#pragma once

#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"

namespace swift_xla {
namespace ir {
namespace ops {

class MaskedFill : public Node {
 public:
  MaskedFill(const Value& input, const Value& mask, at::Scalar value);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  at::Scalar value() const { return value_; }

 private:
  at::Scalar value_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
