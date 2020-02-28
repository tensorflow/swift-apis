#pragma once

#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"

namespace swift_xla {
namespace ir {
namespace ops {

class Softmax : public Node {
 public:
  Softmax(const Value& input, xla::int64 dim,
          c10::optional<at::ScalarType> dtype);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  xla::int64 dim() const { return dim_; }

  const c10::optional<at::ScalarType>& dtype() const { return dtype_; }

 private:
  xla::int64 dim_;
  c10::optional<at::ScalarType> dtype_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
