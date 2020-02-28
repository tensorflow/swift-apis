#ifndef X10_XLA_TENSOR_STRIDED_SLICE_HELPERS_H_
#define X10_XLA_TENSOR_STRIDED_SLICE_HELPERS_H_

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/shape.h"

namespace swift_xla {

// XLA slice parameters and output size for indexing operations.
struct StridedSliceSpec {
  absl::InlinedVector<xla::int64, 4> begin;
  absl::InlinedVector<xla::int64, 4> end;
  absl::InlinedVector<xla::int64, 4> strides;
  absl::InlinedVector<xla::int64, 4> processing_sizes;
  absl::InlinedVector<xla::int64, 4> final_sizes;
};

// Compute the slice parameters and output size to be used when lowering an
// indexing operation.
StridedSliceSpec ComputeIndexingBoundsAndStrides(
    absl::Span<const xla::int64> input_sizes,
    absl::Span<const xla::int64> begin, absl::Span<const xla::int64> end,
    absl::Span<const xla::int64> strides, xla::int32 begin_mask,
    xla::int32 end_mask, xla::int32 ellipsis_mask, xla::int32 new_axis_mask,
    xla::int32 shrink_axis_mask);

}  // namespace swift_xla

#endif  // X10_XLA_TENSOR_STRIDED_SLICE_HELPERS_H_
