#pragma once

#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"
#include "tensorflow/compiler/tf2xla/kernels/conv_op_helpers.h"

namespace swift_xla {
namespace ir {
namespace ops {

class TfConvBackpropInput : public Node {
 public:
  TfConvBackpropInput(std::vector<xla::int64> input_sizes, const Value& filter,
                      const Value& out_backprop, bool depthwise,
                      std::vector<xla::int64> strides,
                      tensorflow::Padding padding,
                      std::vector<xla::int64> explicit_paddings,
                      tensorflow::TensorFormat data_format,
                      std::vector<xla::int64> dilations);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<xla::int64>& input_sizes() const { return input_sizes_; }

  bool depthwise() const { return depthwise_; }

  const std::vector<xla::int64>& strides() const { return strides_; }

  tensorflow::Padding padding() const { return padding_; }

  const std::vector<xla::int64>& explicit_paddings() const {
    return explicit_paddings_;
  }

  tensorflow::TensorFormat data_format() const { return data_format_; }

  const std::vector<xla::int64>& dilations() const { return dilations_; }

 private:
  std::vector<xla::int64> input_sizes_;
  bool depthwise_;
  std::vector<xla::int64> strides_;
  tensorflow::Padding padding_;
  std::vector<xla::int64> explicit_paddings_;
  tensorflow::TensorFormat data_format_;
  std::vector<xla::int64> dilations_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
