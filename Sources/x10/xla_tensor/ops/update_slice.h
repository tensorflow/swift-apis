#pragma once

#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"

namespace swift_xla {
namespace ir {
namespace ops {

class UpdateSlice : public Node {
 public:
  UpdateSlice(const Value& input, const Value& source,
              absl::Span<const xla::int64> base_indices);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<xla::int64>& base_indices() const { return base_indices_; }

 private:
  std::vector<xla::int64> base_indices_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
