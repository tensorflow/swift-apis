#pragma once

#include "tensorflow/compiler/tf2xla/xla_tensor/cross_replica_reduces.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"

namespace swift_xla {
namespace ir {
namespace ops {

class CollectivePermute : public Node {
 public:
  CollectivePermute(
      const Value& input, const Value& token,
      std::vector<std::pair<xla::int64, xla::int64>> source_target_pairs);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const std::vector<std::pair<xla::int64, xla::int64>>& source_target_pairs()
      const {
    return source_target_pairs_;
  }

 private:
  std::vector<std::pair<xla::int64, xla::int64>> source_target_pairs_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
