#pragma once

#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"
#include "tensorflow/compiler/xla/client/lib/pooling.h"
#include "tensorflow/compiler/xla/client/padding.h"

namespace swift_xla {
namespace ir {
namespace ops {

class XlaMaxPool : public Node {
 public:
  XlaMaxPool(const Value& input, std::vector<xla::int64> kernel_size,
             std::vector<xla::int64> strides, xla::Padding padding,
             xla::TensorFormat data_format);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<xla::int64>& kernel_size() const { return kernel_size_; }

  const std::vector<xla::int64>& strides() const { return strides_; }

  xla::Padding padding() const { return padding_; }

  const xla::TensorFormat& data_format() const { return data_format_; }

 private:
  // The parameters of the pooling.
  std::vector<xla::int64> kernel_size_;
  std::vector<xla::int64> strides_;
  xla::Padding padding_;
  xla::TensorFormat data_format_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
