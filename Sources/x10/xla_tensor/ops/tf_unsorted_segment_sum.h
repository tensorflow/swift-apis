#ifndef X10_XLA_TENSOR_OPS_TF_UNSORTED_SEGMENT_SUM_H_
#define X10_XLA_TENSOR_OPS_TF_UNSORTED_SEGMENT_SUM_H_

#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"

namespace swift_xla {
namespace ir {
namespace ops {

class TfUnsortedSegmentSum : public Node {
 public:
  TfUnsortedSegmentSum(const Value& data, const Value& indices,
                       xla::int64 num_segments);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  xla::int64 num_segments() const { return num_segments_; }

 private:
  xla::int64 num_segments_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla

#endif  // X10_XLA_TENSOR_OPS_TF_UNSORTED_SEGMENT_SUM_H_
