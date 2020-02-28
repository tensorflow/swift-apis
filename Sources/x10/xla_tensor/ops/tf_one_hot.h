#pragma once

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"

namespace swift_xla {
namespace ir {
namespace ops {

class TfOneHot : public Node {
 public:
  TfOneHot(const Value& indices, const Value& on_value, const Value& off_value,
           xla::int64 depth, xla::int64 axis);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  xla::int64 depth() const { return depth_; }

  xla::int64 axis() const { return axis_; }

 private:
  xla::int64 depth_;
  xla::int64 axis_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
