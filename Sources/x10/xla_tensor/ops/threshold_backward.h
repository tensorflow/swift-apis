#pragma once

#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"

namespace swift_xla {
namespace ir {
namespace ops {

class ThresholdBackward : public Node {
 public:
  ThresholdBackward(const Value& grad_output, const Value& input,
                    float threshold);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  float threshold() const { return threshold_; }

 private:
  float threshold_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
