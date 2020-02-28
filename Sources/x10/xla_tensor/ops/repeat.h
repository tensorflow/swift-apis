#pragma once

#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"

namespace swift_xla {
namespace ir {
namespace ops {

class Repeat : public Node {
 public:
  Repeat(const Value& input, std::vector<xla::int64> repeats);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<xla::int64>& repeats() const { return repeats_; }

 private:
  // The number of repeats along each dimension.
  std::vector<xla::int64> repeats_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
