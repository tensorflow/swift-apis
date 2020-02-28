#pragma once

#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"

namespace swift_xla {
namespace ir {
namespace ops {

class Token : public Node {
 public:
  Token();

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
