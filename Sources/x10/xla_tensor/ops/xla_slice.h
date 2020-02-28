#ifndef X10_XLA_TENSOR_OPS_XLA_SLICE_H_
#define X10_XLA_TENSOR_OPS_XLA_SLICE_H_

#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"

namespace swift_xla {
namespace ir {
namespace ops {

class XlaSlice : public Node {
 public:
  XlaSlice(const Value& operand, std::vector<xla::int64> start_indices,
           std::vector<xla::int64> limit_indices,
           std::vector<xla::int64> strides);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<xla::int64>& start_indices() const {
    return start_indices_;
  }

  const std::vector<xla::int64>& limit_indices() const {
    return limit_indices_;
  }

  const std::vector<xla::int64>& strides() const { return strides_; }

 private:
  std::vector<xla::int64> start_indices_;
  std::vector<xla::int64> limit_indices_;
  std::vector<xla::int64> strides_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla

#endif  // X10_XLA_TENSOR_OPS_XLA_SLICE_H_
