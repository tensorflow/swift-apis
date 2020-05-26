// Copyright 2020 TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/compiler/tf2xla/xla_tensor/data_ops.h"

#include <algorithm>
#include <functional>
#include <numeric>

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/convert_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/reduction.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/tensor_util.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"

namespace swift_xla {
namespace {

bool IsSparseGather(const xla::Shape& input_shape,
                    const xla::Shape& index_shape, xla::int64 dim) {
  static int dense_gather_factor =
      xla::sys_util::GetEnvInt("XLA_DENSE_GATHER_FACTOR", 100);
  xla::int64 input_elements = xla::ShapeUtil::ElementsIn(input_shape);
  xla::int64 index_elements = xla::ShapeUtil::ElementsIn(index_shape);
  // Simple heuristic. Might need fine tuning.
  return index_elements < input_elements / dense_gather_factor;
}

xla::XlaOp MirrorPadInDimensions(xla::XlaOp input,
                                 absl::Span<const xla::int64> padding,
                                 tensorflow::MirrorPadMode mode) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  XLA_CHECK_GE(2 * input_shape.rank(), padding.size());
  XLA_CHECK_EQ(padding.size() % 2, 0) << "Uneven padding: " << padding.size();

  xla::int64 excluded_edges =
      mode == tensorflow::MirrorPadMode::REFLECT ? 1 : 0;
  xla::XlaOp result = input;
  for (size_t i = 0; i < padding.size(); i += 2) {
    xla::int64 dim = input_shape.rank() - 1 - i / 2;
    xla::int64 dim_size = input_shape.dimensions(dim);
    xla::int64 lhs_padding = padding[i];
    xla::int64 rhs_padding = padding[i + 1];

    XLA_CHECK(lhs_padding >= 0 && lhs_padding <= dim_size - excluded_edges);
    XLA_CHECK(rhs_padding >= 0 && rhs_padding <= dim_size - excluded_edges);

    xla::XlaOp reverse = xla::Rev(result, {dim});
    xla::XlaOp lhs_pad =
        xla::SliceInDim(reverse, dim_size - excluded_edges - lhs_padding,
                        dim_size - excluded_edges, 1, dim);
    xla::XlaOp rhs_pad = xla::SliceInDim(reverse, excluded_edges,
                                         excluded_edges + rhs_padding, 1, dim);
    result = xla::ConcatInDim(input.builder(), {lhs_pad, result, rhs_pad}, dim);
  }
  return result;
}

xla::XlaOp MirrorPadInDimensionsBackward(
    xla::XlaOp grad_output, absl::Span<const xla::int64> input_size,
    absl::Span<const xla::int64> padding, tensorflow::MirrorPadMode mode) {
  const xla::Shape& grad_output_shape = XlaHelpers::ShapeOfXlaOp(grad_output);
  XLA_CHECK_GE(2 * grad_output_shape.rank(), padding.size());
  XLA_CHECK_EQ(padding.size() % 2, 0) << "Uneven padding: " << padding.size();

  xla::int64 excluded_edges =
      mode == tensorflow::MirrorPadMode::REFLECT ? 1 : 0;
  xla::XlaOp grad = grad_output;
  for (size_t i = 0; i < padding.size(); i += 2) {
    xla::int64 dim = grad_output_shape.rank() - 1 - i / 2;
    xla::int64 dim_size = grad_output_shape.dimensions(dim);
    xla::int64 lhs_padding = padding[i];
    xla::int64 rhs_padding = padding[i + 1];

    XLA_CHECK(lhs_padding >= 0 && lhs_padding <= dim_size - excluded_edges);
    XLA_CHECK(rhs_padding >= 0 && rhs_padding <= dim_size - excluded_edges);

    xla::XlaOp lhs_pad = xla::SliceInDim(grad, 0, lhs_padding, 1, dim);
    xla::XlaOp reverse_lhs_pad = xla::Rev(lhs_pad, {dim});
    xla::XlaOp padded_lhs_pad =
        PadInDim(reverse_lhs_pad, dim,
                 /*pad_lo=*/excluded_edges,
                 /*pad_hi=*/input_size[dim] - excluded_edges - lhs_padding);

    xla::XlaOp rhs_pad =
        xla::SliceInDim(grad, dim_size - rhs_padding, dim_size, 1, dim);
    xla::XlaOp reverse_rhs_pad = xla::Rev(rhs_pad, {dim});
    xla::XlaOp padded_rhs_pad =
        PadInDim(reverse_rhs_pad, dim,
                 /*pad_lo=*/input_size[dim] - excluded_edges - rhs_padding,
                 /*pad_hi=*/excluded_edges);

    xla::XlaOp grad_core =
        xla::SliceInDim(grad, lhs_padding, dim_size - rhs_padding, 1, dim);
    grad = padded_lhs_pad + grad_core + padded_rhs_pad;
  }
  return grad;
}

}  // namespace

bool IsSparseGather(xla::XlaOp input, xla::XlaOp index, xla::int64 dim) {
  return IsSparseGather(XlaHelpers::ShapeOfXlaOp(input),
                        XlaHelpers::ShapeOfXlaOp(index), dim);
}

std::vector<xla::int64> GetCompleteShape(
    absl::Span<const xla::int64> output_sizes,
    absl::Span<const xla::int64> input_sizes) {
  c10::optional<size_t> incomplete_dim;
  xla::int64 incomplete_element_count = 1;
  for (size_t dim = 0; dim < output_sizes.size(); ++dim) {
    xla::int64 dim_size = output_sizes[dim];
    if (dim_size < 0) {
      XLA_CHECK(!incomplete_dim)
          << "More than one incomplete dimension found: " << *incomplete_dim
          << " and " << dim;
      incomplete_dim = dim;
    } else {
      incomplete_element_count *= dim_size;
    }
  }
  xla::int64 total_element_count = xla::util::Multiply<xla::int64>(input_sizes);
  if (!incomplete_dim) {
    XLA_CHECK_EQ(total_element_count,
                 xla::util::Multiply<xla::int64>(output_sizes))
        << "(" << absl::StrJoin(output_sizes, ", ") << ") vs. ("
        << absl::StrJoin(input_sizes, ", ") << ")";
    return xla::util::ToVector<xla::int64>(output_sizes);
  }
  XLA_CHECK_GT(incomplete_element_count, 0)
      << "Cannot reshape tensor of 0 elements into shape "
      << "(" << absl::StrJoin(output_sizes, ", ")
      << ") because the unspecified dimension size -1 can be any value";
  XLA_CHECK_EQ(total_element_count % incomplete_element_count, 0)
      << "(" << absl::StrJoin(output_sizes, ", ") << ") vs. ("
      << absl::StrJoin(input_sizes, ", ") << ")";
  std::vector<xla::int64> complete_output_sizes =
      xla::util::ToVector<xla::int64>(output_sizes);
  complete_output_sizes[*incomplete_dim] =
      total_element_count / incomplete_element_count;
  return complete_output_sizes;
}

xla::XlaOp BuildView(xla::XlaOp input,
                     absl::Span<const xla::int64> output_sizes) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  const auto complete_output_sizes =
      GetCompleteShape(output_sizes, input_shape.dimensions());
  return XlaHelpers::DynamicReshape(input, complete_output_sizes);
}

xla::XlaOp SqueezeTrivialDimension(xla::XlaOp input, xla::int64 dim) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  XLA_CHECK_LT(dim, input_shape.rank());
  if (input_shape.dimensions(dim) != 1) {
    return input;
  }
  auto output_sizes = BuildSqueezedDimensions(input_shape.dimensions(), dim);
  return XlaHelpers::DynamicReshape(input, output_sizes);
}

xla::XlaOp SqueezeAllTrivialDimensions(xla::XlaOp input) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  auto output_sizes =
      BuildSqueezedDimensions(input_shape.dimensions(), /*squeeze_dim=*/-1);
  return XlaHelpers::DynamicReshape(input, output_sizes);
}

xla::XlaOp BuildExpand(xla::XlaOp input,
                       absl::Span<const xla::int64> output_sizes) {
  auto input_sizes = XlaHelpers::SizesOfXlaOp(input);
  // Adjust the rank of the input to match the rank of the output.
  XLA_CHECK_LE(input_sizes.size(), output_sizes.size());
  input_sizes.insert(input_sizes.begin(),
                     output_sizes.size() - input_sizes.size(), 1);
  xla::XlaOp implicit_reshape = XlaHelpers::DynamicReshape(input, input_sizes);
  return xla::BroadcastInDim(implicit_reshape, output_sizes,
                             xla::util::Iota<xla::int64>(output_sizes.size()));
}

std::vector<xla::int64> BuildSqueezedDimensions(
    absl::Span<const xla::int64> dimensions, xla::int64 squeeze_dim) {
  std::vector<xla::int64> output_dimensions;
  for (xla::int64 i = 0; i < dimensions.size(); ++i) {
    xla::int64 dim = dimensions[i];
    if (dim != 1 || (i != squeeze_dim && squeeze_dim >= 0)) {
      output_dimensions.push_back(dim);
    }
  }
  return output_dimensions;
}

std::vector<xla::int64> BuildUnsqueezeDimensions(
    absl::Span<const xla::int64> dimensions, xla::int64 dim) {
  XLA_CHECK_LE(dim, dimensions.size());
  auto unsqueeze_dimensions = xla::util::ToVector<xla::int64>(dimensions);
  unsqueeze_dimensions.insert(unsqueeze_dimensions.begin() + dim, 1);
  return unsqueeze_dimensions;
}

xla::XlaOp BuildUnsqueeze(xla::XlaOp input, xla::int64 dim) {
  auto dimensions =
      BuildUnsqueezeDimensions(XlaHelpers::SizesOfXlaOp(input), dim);
  return XlaHelpers::DynamicReshape(input, dimensions);
}

xla::XlaOp BuildStack(absl::Span<const xla::XlaOp> inputs, xla::int64 dim) {
  // Reshape inputs along the dim axis.
  XLA_CHECK_GT(inputs.size(), 0);
  std::vector<xla::XlaOp> reshaped_inputs;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto input_size = XlaHelpers::SizesOfXlaOp(inputs[i]);
    input_size.insert(input_size.begin() + dim, 1);
    reshaped_inputs.push_back(
        XlaHelpers::DynamicReshape(inputs[i], input_size));
  }
  return xla::ConcatInDim(inputs[0].builder(), reshaped_inputs, dim);
}

xla::XlaOp BuildCat(absl::Span<const xla::XlaOp> inputs, xla::int64 dim) {
  XLA_CHECK_GT(inputs.size(), 0);
  return xla::ConcatInDim(inputs[0].builder(), inputs, dim);
}

xla::XlaOp BuildRepeat(xla::XlaOp input, absl::Span<const xla::int64> repeats) {
  const auto input_sizes = XlaHelpers::SizesOfXlaOp(input);
  XLA_CHECK_GE(repeats.size(), input_sizes.size())
      << "Number of dimensions of repeat dims can not be smaller than number "
         "of dimensions of tensor";
  size_t broadcast_dims = repeats.size() - input_sizes.size();
  xla::XlaOp repeated = input;
  for (size_t dim = 0; dim < input_sizes.size(); ++dim) {
    std::vector<xla::XlaOp> repeated_inputs(repeats[broadcast_dims + dim],
                                            repeated);
    repeated = xla::ConcatInDim(input.builder(), repeated_inputs, dim);
  }
  if (repeats.size() > input_sizes.size()) {
    std::vector<xla::int64> remaining_repeats(repeats.begin(),
                                              repeats.begin() + broadcast_dims);
    repeated = xla::Broadcast(repeated, remaining_repeats);
  }
  return repeated;
}

size_t ComputeSplitCount(xla::int64 dim_size,
                         absl::Span<const xla::int64> split_sizes) {
  size_t count = 0;
  for (auto size : split_sizes) {
    if (size > dim_size) {
      break;
    }
    dim_size -= size;
    ++count;
  }
  return count;
}

std::vector<xla::XlaOp> BuildSplit(xla::XlaOp input,
                                   absl::Span<const xla::int64> split_sizes,
                                   xla::int64 dim) {
  const auto input_sizes = XlaHelpers::SizesOfXlaOp(input);
  xla::int64 dim_size = input_sizes.at(dim);
  xla::int64 index = 0;
  std::vector<xla::XlaOp> splits;
  for (auto size : split_sizes) {
    if (index + size > dim_size) {
      break;
    }
    splits.emplace_back(xla::SliceInDim(input, index, index + size, 1, dim));
    index += size;
  }
  return splits;
}

xla::XlaOp BuildUpdateSlice(xla::XlaOp input, xla::XlaOp source,
                            absl::Span<const xla::int64> base_indices) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  const xla::Shape& source_shape = XlaHelpers::ShapeOfXlaOp(source);
  xla::XlaOp update_source = source;
  if (source_shape.element_type() != input_shape.element_type()) {
    update_source = ConvertTo(source, source_shape.element_type(),
                              input_shape.element_type(), /*device=*/nullptr);
  }
  xla::XlaOp reshaped_source =
      XlaHelpers::ReshapeToRank(update_source, input_shape.rank());
  std::vector<xla::XlaOp> start_indices;
  for (auto index : base_indices) {
    start_indices.push_back(
        XlaHelpers::ScalarValue<xla::int64>(index, input.builder()));
  }
  return xla::DynamicUpdateSlice(input, reshaped_source, start_indices);
}

xla::XlaOp BuildSlice(xla::XlaOp input,
                      absl::Span<const xla::int64> base_indices,
                      absl::Span<const xla::int64> sizes) {
  XLA_CHECK_EQ(base_indices.size(), sizes.size());
  std::vector<xla::int64> limit_indices(base_indices.begin(),
                                        base_indices.end());
  std::transform(limit_indices.begin(), limit_indices.end(), sizes.begin(),
                 limit_indices.begin(), std::plus<xla::int64>());
  std::vector<xla::int64> strides(base_indices.size(), 1);
  return xla::Slice(input, base_indices, limit_indices, strides);
}

xla::XlaOp BoundIndices(xla::XlaOp index, xla::XlaOp max_index) {
  const xla::Shape& index_shape = XlaHelpers::ShapeOfXlaOp(index);
  return xla::Select(
      xla::Ge(index, xla::Zero(index.builder(), index_shape.element_type())),
      index, index + max_index);
}

xla::XlaOp BuildTake(xla::XlaOp input, xla::XlaOp index) {
  static const int take_dim = 0;
  xla::Shape input_shape;
  xla::XlaOp r1_input = XlaHelpers::Flatten(input, &input_shape);
  xla::Shape index_shape;
  xla::XlaOp r1_index = XlaHelpers::Flatten(index, &index_shape);
  xla::XlaOp max_index =
      XlaHelpers::ScalarValue(xla::ShapeUtil::ElementsIn(input_shape),
                              index_shape.element_type(), index.builder());
  xla::XlaOp bound_index = BoundIndices(r1_index, max_index);
  xla::XlaOp r1_result =
      xla::TorchGather(r1_input, bound_index, take_dim,
                       IsSparseGather(input_shape, index_shape, take_dim));
  return XlaHelpers::DynamicReshape(r1_result, index_shape.dimensions());
}

xla::XlaOp BuildResize(xla::XlaOp input, absl::Span<const xla::int64> size) {
  xla::Shape input_shape;
  xla::XlaOp r1_input = XlaHelpers::Flatten(input, &input_shape);
  xla::int64 num_elements = xla::ShapeUtil::ElementsIn(input_shape);
  xla::int64 new_num_elements = xla::util::Multiply<xla::int64>(size);
  xla::XlaOp resized_input = input;
  if (num_elements > new_num_elements) {
    resized_input = xla::SliceInDim(r1_input, 0, new_num_elements, 1, 0);
  } else if (new_num_elements > num_elements) {
    xla::XlaOp zero = xla::Zero(input.builder(), input_shape.element_type());
    xla::PaddingConfig padding_config;
    auto* dims = padding_config.add_dimensions();
    dims->set_edge_padding_low(0);
    dims->set_interior_padding(0);
    dims->set_edge_padding_high(new_num_elements - num_elements);
    resized_input = xla::Pad(r1_input, zero, padding_config);
  }
  return XlaHelpers::DynamicReshape(resized_input, size);
}

xla::XlaOp BuildUnselect(xla::XlaOp target, xla::XlaOp source, xla::int64 dim,
                         xla::int64 start, xla::int64 end, xla::int64 stride) {
  const xla::Shape& target_shape = XlaHelpers::ShapeOfXlaOp(target);
  const xla::Shape& source_shape = XlaHelpers::ShapeOfXlaOp(source);
  if (target_shape.dimensions(dim) == source_shape.dimensions(dim)) {
    // Shortcut for unselects which are fully covering selects.
    XLA_CHECK_EQ(start, 0);
    XLA_CHECK_EQ(stride, 1);
    XLA_CHECK_EQ(end, target_shape.dimensions(dim));
    return source;
  }

  xla::PrimitiveType pred_type =
      GetDevicePrimitiveType(xla::PrimitiveType::PRED, /*device=*/nullptr);
  xla::XlaOp source_true = XlaHelpers::ScalarBroadcast(
      1, pred_type, source_shape.dimensions(), source.builder());
  xla::XlaOp pred_zero = xla::Zero(target.builder(), pred_type);
  xla::XlaOp zero = xla::Zero(target.builder(), target_shape.element_type());
  xla::PaddingConfig padding_config;
  for (xla::int64 i = 0; i < target_shape.rank(); ++i) {
    auto* dims = padding_config.add_dimensions();
    if (i == dim) {
      dims->set_edge_padding_low(start);
      dims->set_interior_padding(stride - 1);

      xla::int64 size = start + source_shape.dimensions(i) +
                        (source_shape.dimensions(i) - 1) * (stride - 1);
      dims->set_edge_padding_high(target_shape.dimensions(i) - size);
    } else {
      XLA_CHECK_EQ(target_shape.dimensions(i), source_shape.dimensions(i))
          << target_shape << " vs. " << source_shape;
      dims->set_edge_padding_low(0);
      dims->set_interior_padding(0);
      dims->set_edge_padding_high(0);
    }
  }
  xla::XlaOp padded_source = xla::Pad(source, zero, padding_config);
  xla::XlaOp mask = xla::Pad(source_true, pred_zero, padding_config);
  return xla::Select(mask, padded_source, target);
}

xla::XlaOp BuildMirrorPad(xla::XlaOp input,
                          absl::Span<const xla::int64> padding,
                          tensorflow::MirrorPadMode mode) {
  return MirrorPadInDimensions(input, padding, mode);
}

xla::XlaOp BuildMirrorPadBackward(xla::XlaOp grad_output,
                                  absl::Span<const xla::int64> input_size,
                                  absl::Span<const xla::int64> padding,
                                  tensorflow::MirrorPadMode mode) {
  const xla::Shape& grad_output_shape = XlaHelpers::ShapeOfXlaOp(grad_output);
  std::vector<xla::int64> spatial_dims;
  for (xla::int64 dim = grad_output_shape.rank() - 1; dim >= 0; --dim) {
    spatial_dims.push_back(dim);
  }
  return MirrorPadInDimensionsBackward(grad_output, input_size, padding, mode);
}

xla::XlaOp BuildReflectionPad2d(xla::XlaOp input,
                                absl::Span<const xla::int64> padding) {
  return MirrorPadInDimensions(input, padding,
                               tensorflow::MirrorPadMode::REFLECT);
}

xla::XlaOp BuildReflectionPadBackward(xla::XlaOp grad_output, xla::XlaOp input,
                                      absl::Span<const xla::int64> padding) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  return MirrorPadInDimensionsBackward(grad_output, input_shape.dimensions(),
                                       padding,
                                       tensorflow::MirrorPadMode::REFLECT);
}

xla::XlaOp BuildReplicationPad(xla::XlaOp input,
                               absl::Span<const xla::int64> padding) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  XLA_CHECK_GE(2 * input_shape.rank(), padding.size());
  XLA_CHECK_EQ(padding.size() % 2, 0) << "Uneven padding: " << padding.size();
  xla::XlaOp result = input;
  for (size_t i = 0; i < padding.size(); i += 2) {
    xla::int64 dim = input_shape.rank() - 1 - i / 2;
    if ((padding[i] != 0 || padding[i + 1] != 0) &&
        input_shape.dimensions(dim) > 0) {
      std::vector<xla::XlaOp> parts;
      if (padding[i] != 0) {
        xla::XlaOp pad1 = xla::SliceInDim(result, 0, 1, 1, dim);
        parts.push_back(
            XlaHelpers::BroadcastDimensions(pad1, {dim}, {padding[i]}));
      }
      parts.push_back(result);
      if (padding[i + 1] != 0) {
        xla::XlaOp pad1 =
            xla::SliceInDim(result, input_shape.dimensions(dim) - 1,
                            input_shape.dimensions(dim), 1, dim);
        parts.push_back(
            XlaHelpers::BroadcastDimensions(pad1, {dim}, {padding[i + 1]}));
      }
      result = xla::ConcatInDim(result.builder(), parts, dim);
    }
  }
  return result;
}

xla::XlaOp BuildReplicationPadBackward(xla::XlaOp grad_output, xla::XlaOp input,
                                       absl::Span<const xla::int64> padding) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  const xla::Shape& grad_output_shape = XlaHelpers::ShapeOfXlaOp(grad_output);
  XLA_CHECK_GE(2 * grad_output_shape.rank(), padding.size());
  XLA_CHECK_EQ(padding.size() % 2, 0) << "Uneven padding: " << padding.size();

  xla::XlaOp grad = grad_output;
  for (size_t i = 0; i < padding.size(); i += 2) {
    xla::int64 dim = grad_output_shape.rank() - 1 - i / 2;
    xla::int64 dim_size = grad_output_shape.dimensions(dim);
    xla::int64 lhs_padding = padding[i];
    xla::int64 rhs_padding = padding[i + 1];

    XLA_CHECK(lhs_padding >= 0 && lhs_padding <= dim_size - 1);
    XLA_CHECK(rhs_padding >= 0 && rhs_padding <= dim_size - 1);

    xla::XlaOp lhs_pad = xla::SliceInDim(grad, 0, lhs_padding, 1, dim);
    xla::XlaOp reduced_lhs_pad =
        BuildSum(lhs_pad, {dim}, /*keep_reduced_dimensions=*/true);
    xla::XlaOp padded_lhs_pad =
        PadInDim(reduced_lhs_pad, dim,
                 /*pad_lo=*/0,
                 /*pad_hi=*/input_shape.dimensions(dim) - 1);

    xla::XlaOp rhs_pad =
        xla::SliceInDim(grad, dim_size - rhs_padding, dim_size, 1, dim);
    xla::XlaOp reduced_rhs_pad =
        BuildSum(rhs_pad, {dim}, /*keep_reduced_dimensions=*/true);
    xla::XlaOp padded_rhs_pad =
        PadInDim(reduced_rhs_pad, dim,
                 /*pad_lo=*/input_shape.dimensions(dim) - 1,
                 /*pad_hi=*/0);

    xla::XlaOp grad_core =
        xla::SliceInDim(grad, lhs_padding, dim_size - rhs_padding, 1, dim);
    grad = padded_lhs_pad + grad_core + padded_rhs_pad;
  }
  return grad;
}

xla::XlaOp PadInDim(xla::XlaOp input, xla::int64 dim, xla::int64 pad_lo,
                    xla::int64 pad_hi, const xla::XlaOp* pad_value) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp zero;
  if (pad_value == nullptr) {
    zero = xla::Zero(input.builder(), input_shape.element_type());
    pad_value = &zero;
  }
  xla::PaddingConfig padding_config;
  for (xla::int64 i = 0; i < input_shape.rank(); ++i) {
    auto* dims = padding_config.add_dimensions();
    dims->set_interior_padding(0);
    if (i == dim) {
      dims->set_edge_padding_low(pad_lo);
      dims->set_edge_padding_high(pad_hi);
    } else {
      dims->set_edge_padding_low(0);
      dims->set_edge_padding_high(0);
    }
  }
  return xla::Pad(input, *pad_value, padding_config);
}

}  // namespace swift_xla
