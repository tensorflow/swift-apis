#pragma once

#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"

namespace swift_xla {
namespace ir {
namespace ops {

class ConstantPadNd : public Node {
 public:
  ConstantPadNd(const Value& input, std::vector<xla::int64> pad,
                at::Scalar value);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  at::Scalar value() const { return value_; }

  const std::vector<xla::int64> pad() const { return pad_; }

 private:
  std::vector<xla::int64> pad_;
  at::Scalar value_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
