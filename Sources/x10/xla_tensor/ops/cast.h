#pragma once

#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"

namespace swift_xla {
namespace ir {
namespace ops {

class Cast : public Node {
 public:
  Cast(const Value& input, at::ScalarType dtype);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  at::ScalarType dtype() const { return dtype_; }

 private:
  at::ScalarType dtype_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
