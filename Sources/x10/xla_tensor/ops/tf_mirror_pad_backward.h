#ifndef X10_XLA_TENSOR_OPS_TF_MIRROR_PAD_BACKWARD_H_
#define X10_XLA_TENSOR_OPS_TF_MIRROR_PAD_BACKWARD_H_

#include <vector>

#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"
#include "tensorflow/core/util/mirror_pad_mode.h"

namespace swift_xla {
namespace ir {
namespace ops {

class TfMirrorPadBackward : public Node {
 public:
  TfMirrorPadBackward(const Value& grad_output,
                      std::vector<xla::int64> input_size,
                      std::vector<xla::int64> padding,
                      tensorflow::MirrorPadMode mode);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<xla::int64>& input_size() const { return input_size_; }

  const std::vector<xla::int64>& padding() const { return padding_; }

  tensorflow::MirrorPadMode mode() const { return mode_; }

 private:
  std::vector<xla::int64> input_size_;
  std::vector<xla::int64> padding_;
  tensorflow::MirrorPadMode mode_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla

#endif  // X10_XLA_TENSOR_OPS_TF_MIRROR_PAD_BACKWARD_H_
