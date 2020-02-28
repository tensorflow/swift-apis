#pragma once

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"
#include "tensorflow/compiler/xla/client/lib/pooling.h"

namespace swift_xla {
namespace ir {
namespace ops {

class XlaAvgPoolGrad : public Node {
 public:
  XlaAvgPoolGrad(const Value& out_backprop,
                 std::vector<xla::int64> gradients_size,
                 std::vector<xla::int64> kernel_size,
                 std::vector<xla::int64> stride,
                 std::vector<std::pair<xla::int64, xla::int64>> spatial_padding,
                 const xla::TensorFormat& data_format,
                 const bool counts_include_padding);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<xla::int64>& gradients_size() const {
    return gradients_size_;
  }

  const std::vector<xla::int64>& kernel_size() const { return kernel_size_; }

  const std::vector<xla::int64>& stride() const { return stride_; }

  const std::vector<std::pair<xla::int64, xla::int64>>& spatial_padding()
      const {
    return spatial_padding_;
  }

  const xla::TensorFormat& data_format() const { return data_format_; }

  bool counts_include_padding() const { return counts_include_padding_; }

 private:
  // Size of input gradients.
  std::vector<xla::int64> gradients_size_;
  // The parameters of the pooling.
  std::vector<xla::int64> kernel_size_;
  std::vector<xla::int64> stride_;
  std::vector<std::pair<xla::int64, xla::int64>> spatial_padding_;
  xla::TensorFormat data_format_;
  // Whether the counts used to compute the average should include the added
  // padding.
  bool counts_include_padding_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
