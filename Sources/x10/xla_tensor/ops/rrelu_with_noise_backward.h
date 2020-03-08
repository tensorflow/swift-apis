#pragma once

#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"

namespace swift_xla {
namespace ir {
namespace ops {

class RreluWithNoiseBackward : public Node {
 public:
  RreluWithNoiseBackward(const Value& grad_output, const Value& input,
                         const Value& noise, at::Scalar lower, at::Scalar upper,
                         bool training);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const at::Scalar& lower() const { return lower_; }

  const at::Scalar& upper() const { return upper_; }

  bool training() const { return training_; }

 private:
  at::Scalar lower_;
  at::Scalar upper_;
  bool training_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla