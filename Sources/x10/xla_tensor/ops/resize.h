#pragma once

#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"

namespace swift_xla {
namespace ir {
namespace ops {

class Resize : public Node {
 public:
  Resize(const Value& input, std::vector<xla::int64> size);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<xla::int64>& size() const { return size_; }

 private:
  std::vector<xla::int64> size_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
