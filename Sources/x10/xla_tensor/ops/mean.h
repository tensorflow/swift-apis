#pragma once

#include <vector>

#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"
#include "tensorflow/compiler/xla/types.h"

namespace swift_xla {
namespace ir {
namespace ops {

class Mean : public Node {
 public:
  Mean(const Value& input, std::vector<xla::int64> dimensions,
       bool keep_reduced_dimensions, c10::optional<at::ScalarType> dtype);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const std::vector<xla::int64>& dimensions() const { return dimensions_; }

  bool keep_reduced_dimensions() const { return keep_reduced_dimensions_; }

  const c10::optional<at::ScalarType>& dtype() const { return dtype_; }

 private:
  std::vector<xla::int64> dimensions_;
  bool keep_reduced_dimensions_;
  c10::optional<at::ScalarType> dtype_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
