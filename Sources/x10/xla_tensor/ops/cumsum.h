#pragma once

#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"

namespace swift_xla {
namespace ir {
namespace ops {

class CumSum : public Node {
 public:
  CumSum(const Value& input, xla::int64 dim,
         c10::optional<at::ScalarType> dtype, bool exclusive, bool reverse);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  xla::int64 dim() const { return dim_; }

  const c10::optional<at::ScalarType>& dtype() const { return dtype_; }

 private:
  xla::int64 dim_;
  c10::optional<at::ScalarType> dtype_;
  bool exclusive_;
  bool reverse_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
