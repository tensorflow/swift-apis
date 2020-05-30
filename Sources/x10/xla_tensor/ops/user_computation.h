#pragma once

#include "tensorflow/compiler/tf2xla/xla_tensor/computation.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"

namespace swift_xla {
namespace ir {
namespace ops {

class UserComputation : public Node {
 public:
  UserComputation(OpKind op, OpList operands, ComputationPtr computation);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const ComputationPtr& computation() const { return computation_; }

 private:
  ComputationPtr computation_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
