#pragma once

#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/reduction.h"
#include "tensorflow/compiler/xla/types.h"

namespace swift_xla {
namespace ir {
namespace ops {

class L1Loss : public Node {
 public:
  L1Loss(const Value& input, const Value& target, ReductionMode reduction);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  ReductionMode reduction() const { return reduction_; }

 private:
  ReductionMode reduction_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
