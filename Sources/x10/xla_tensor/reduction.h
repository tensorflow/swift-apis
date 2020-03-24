/*
 * Copyright 2020 TensorFlow Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace swift_xla {

enum class ReductionMode {
  kNone,
  kMean,
  kSum,
};

xla::XlaOp BuildBinaryCrossEntropy(xla::XlaOp input, xla::XlaOp target,
                                   const absl::optional<xla::XlaOp>& weight,
                                   ReductionMode reduction);

xla::XlaOp BuildBinaryCrossEntropyBackward(
    xla::XlaOp grad_output, xla::XlaOp input, xla::XlaOp target,
    const absl::optional<xla::XlaOp>& weight, ReductionMode reduction);

xla::XlaOp BuildL1Loss(xla::XlaOp input, xla::XlaOp target,
                       ReductionMode reduction);

xla::XlaOp BuildL1LossBackward(xla::XlaOp grad_output, xla::XlaOp input,
                               xla::XlaOp target, ReductionMode reduction);

xla::XlaOp BuildMseLoss(xla::XlaOp input, xla::XlaOp target,
                        ReductionMode reduction);

xla::XlaOp BuildMseLossBackward(xla::XlaOp grad_output, xla::XlaOp input,
                                xla::XlaOp target, ReductionMode reduction);

// Builds a mean by reducing all the dimensions listed in dimensions. If
// keep_reduced_dimensions is true, the reduced dimensions will be retained,
// with value 1.
xla::XlaOp BuildMean(xla::XlaOp input, absl::Span<const xla::int64> dimensions,
                     bool keep_reduced_dimensions);

xla::XlaOp BuildStdDeviation(xla::XlaOp input,
                             absl::Span<const xla::int64> dimensions,
                             bool keep_reduced_dimensions, bool unbiased);

// Builds the sum of all values by reducing all the dimensions listed in
// dimensions. If keep_reduced_dimensions is true, the reduced dimensions will
// be retained, with value 1.
xla::XlaOp BuildSum(xla::XlaOp input, absl::Span<const xla::int64> dimensions,
                    bool keep_reduced_dimensions);

// Builds the max of all values by reducing in the given dimension. If
// keep_reduced_dimensions is true, the reduced dimension will be retained, with
// value 1.
xla::XlaOp BuildMaxInDim(xla::XlaOp input, xla::int64 dim,
                         bool keep_reduced_dimensions);

// Builds the max of all values by reducing in the given dimensions. If
// keep_reduced_dimensions is true, the reduced dimension will be retained, with
// value 1.
xla::XlaOp BuildMaxInDims(xla::XlaOp input,
                          absl::Span<const xla::int64> dimensions,
                          bool keep_reduced_dimensions);

// Builds the min of all values by reducing in the given dimension. If
// keep_reduced_dimensions is true, the reduced dimension will be retained, with
// value 1.
xla::XlaOp BuildMinInDim(xla::XlaOp input, xla::int64 dim,
                         bool keep_reduced_dimensions);

// Compute the indices of the maximum values of a tensor across a dimension.
xla::XlaOp BuildArgMax(xla::XlaOp input, xla::int64 dim, bool keepdim);

// Compute the indices of the minimum values of a tensor across a dimension.
xla::XlaOp BuildArgMin(xla::XlaOp input, xla::int64 dim, bool keepdim);

// Builds the product of all values by reducing all the dimensions listed in
// dimensions. If keep_reduced_dimensions is true, the reduced dimensions will
// be retained, with value 1.
xla::XlaOp BuildProd(xla::XlaOp input, absl::Span<const xla::int64> dimensions,
                     bool keep_reduced_dimensions);

// Compute the cumulative computation specified by "reducer" and "init" in the
// given dimension "dim".
xla::XlaOp BuildCumulativeComputation(xla::XlaOp input, xla::int64 dim,
                                      const xla::XlaComputation& reducer,
                                      xla::XlaOp init, bool exclusive,
                                      bool reverse);

xla::XlaOp BuildAll(xla::XlaOp input, absl::Span<const xla::int64> dimensions,
                    bool keep_reduced_dimensions);

xla::XlaOp BuildAny(xla::XlaOp input, absl::Span<const xla::int64> dimensions,
                    bool keep_reduced_dimensions);

xla::XlaOp BuildLogsumexp(xla::XlaOp input,
                          absl::Span<const xla::int64> dimensions,
                          bool keep_reduced_dimensions);

}  // namespace swift_xla
