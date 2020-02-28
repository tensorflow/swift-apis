#pragma once

#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"

namespace swift_xla {
namespace ir {
namespace ops {

class QR : public Node {
 public:
  QR(const Value& input, bool some);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  bool some() const { return some_; }

 private:
  bool some_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
