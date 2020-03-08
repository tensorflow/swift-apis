#pragma once

#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace swift_xla {
namespace ir {
namespace ops {

// IR node for 2D & 3D convolutions with or without bias.
class ConvolutionOverrideable : public Node {
 public:
  ConvolutionOverrideable(const Value& input, const Value& weight,
                          const Value& bias, std::vector<xla::int64> stride,
                          std::vector<xla::int64> padding,
                          std::vector<xla::int64> dilation, bool transposed,
                          std::vector<xla::int64> output_padding,
                          xla::int64 groups);

  ConvolutionOverrideable(const Value& input, const Value& weight,
                          std::vector<xla::int64> stride,
                          std::vector<xla::int64> padding,
                          std::vector<xla::int64> dilation, bool transposed,
                          std::vector<xla::int64> output_padding,
                          xla::int64 groups);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<xla::int64>& stride() const { return stride_; }

  const std::vector<xla::int64>& padding() const { return padding_; }

  const std::vector<xla::int64>& dilation() const { return dilation_; }

  bool transposed() const { return transposed_; }

  const std::vector<xla::int64>& output_padding() const {
    return output_padding_;
  }

  xla::int64 groups() const { return groups_; }

 private:
  std::vector<xla::int64> stride_;
  std::vector<xla::int64> padding_;
  std::vector<xla::int64> dilation_;
  std::vector<xla::int64> output_padding_;
  bool transposed_;
  xla::int64 groups_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla