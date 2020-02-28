#pragma once

#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"

namespace swift_xla {
namespace ir {
namespace ops {

class Constant : public Node {
 public:
  Constant(xla::Literal value);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const xla::Literal& value() const { return value_; }

 private:
  xla::Literal value_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
