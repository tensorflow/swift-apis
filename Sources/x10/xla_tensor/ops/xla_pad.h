#ifndef X10_XLA_TENSOR_OPS_XLA_PAD_H_
#define X10_XLA_TENSOR_OPS_XLA_PAD_H_

#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"

namespace swift_xla {
namespace ir {
namespace ops {

class XlaPad : public Node {
 public:
  XlaPad(const Value& operand, const Value& padding_value,
         xla::PaddingConfig padding_config);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const xla::PaddingConfig& padding_config() const { return padding_config_; }

 private:
  xla::PaddingConfig padding_config_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla

#endif  // X10_XLA_TENSOR_OPS_XLA_PAD_H_
