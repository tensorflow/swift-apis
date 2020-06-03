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

#include <vector>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/xla_client/device.h"

namespace swift_xla {

xla::XlaOp PadToSize(xla::XlaOp input, absl::Span<const xla::int64> size,
                     absl::optional<xla::XlaOp> pad_value = absl::nullopt);

std::vector<xla::XlaOp> CreateKthValue(xla::XlaOp input, xla::int64 k,
                                       xla::int64 dim, bool keepdim);

std::vector<xla::XlaOp> CreateTopK(xla::XlaOp input, xla::int64 k,
                                   xla::int64 dim, bool largest, bool sorted);

xla::XlaOp CreateMatMul(xla::XlaOp lhs, xla::XlaOp rhs);

xla::XlaOp BuildGer(xla::XlaOp lhs, xla::XlaOp rhs);

xla::XlaOp BuildMatMul(xla::XlaOp lhs, xla::XlaOp rhs, xla::XlaOp bias);

xla::XlaOp BuildDot(xla::XlaOp lhs, xla::XlaOp rhs);

xla::XlaOp BuildBernoulli(xla::XlaOp probability, xla::XlaOp seed,
                          xla::PrimitiveType type);

xla::XlaOp BuildExponential(xla::XlaOp lambda, xla::XlaOp seed,
                            xla::PrimitiveType type);

xla::XlaOp BuildDropout(xla::XlaOp input, float probability, xla::XlaOp seed);

std::vector<xla::XlaOp> CreateBroadcastTensors(
    absl::Span<const xla::XlaOp> operands);

// Similar to tf.gather_nd, used to implement advanced indexing.
xla::XlaOp CreateIndex(xla::XlaOp input, xla::XlaOp indices,
                       xla::int64 start_dim);

// Similar to tf.scatter_nd, used to implement advanced indexing updates.
xla::XlaOp CreateIndexUpdate(
    xla::XlaOp buffer, xla::XlaOp indices, xla::int64 start_dim,
    xla::XlaOp updates,
    const std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp)>& combiner);

xla::XlaOp CreateIndexAdd(xla::XlaOp buffer, xla::int64 dim, xla::XlaOp index,
                          xla::XlaOp value);

xla::XlaOp CreateIndexCopy(xla::XlaOp buffer, xla::int64 dim, xla::XlaOp index,
                           xla::XlaOp value);

xla::XlaOp CreateIndexFill(xla::XlaOp buffer, xla::int64 dim, xla::XlaOp index,
                           xla::XlaOp values);

using XlaOpCombiner = std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp)>;

XlaOpCombiner NumericAddCombiner();

// Used to lower scatter and scatter_add.
xla::XlaOp CreateScatter(const Device& device, xla::XlaOp input,
                         xla::XlaOp index, xla::XlaOp source, xla::int64 dim,
                         const XlaOpCombiner& combiner);

xla::XlaOp CreatePut(const Device& device, xla::XlaOp input, xla::XlaOp index,
                     xla::XlaOp source, bool accumulate);

std::vector<xla::XlaOp> BuildNonZero(xla::XlaOp input);

std::vector<xla::XlaOp> BuildMaskedSelect(xla::XlaOp input, xla::XlaOp mask);

xla::XlaOp BuildMaskedScatter(xla::XlaOp input, xla::XlaOp mask,
                              xla::XlaOp source);

}  // namespace swift_xla
