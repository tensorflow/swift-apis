#pragma once

#include "absl/types/optional.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/reduction.h"

namespace swift_xla {
namespace ir {
namespace ops {

class NllLoss : public Node {
 public:
  NllLoss(const Value& logits, const Value& labels,
          const absl::optional<Value>& weight, ReductionMode reduction,
          int ignore_index);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  ReductionMode reduction() const { return reduction_; }

  int ignore_index() const { return ignore_index_; }

 private:
  ReductionMode reduction_;
  int ignore_index_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
