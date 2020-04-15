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

#include "tensorflow/compiler/tf2xla/xla_tensor/reduction.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/tensor.h"

// Certain tensor operations can be expressed in terms of other tensor
// operations. Add their implementations here instead of the main XLATensor
// class.

namespace swift_xla {
namespace tensor_ops {

XLATensor Cross(const XLATensor& input, const XLATensor& other,
                c10::optional<xla::int64> dim);

XLATensor KlDivBackward(const XLATensor& grad_output, const XLATensor& input,
                        const XLATensor& target, ReductionMode reduction,
                        bool log_target);

XLATensor MakeMatrixWithDiagonal(const XLATensor& input, xla::int64 diagonal);

XLATensor SmoothL1Loss(const XLATensor& input, const XLATensor& target,
                       ReductionMode reduction);

XLATensor SmoothL1LossBackward(const XLATensor& grad_output,
                               const XLATensor& input, const XLATensor& target,
                               ReductionMode reduction);

XLATensor Softplus(const XLATensor& input, at::Scalar beta,
                   at::Scalar threshold);

XLATensor SoftplusBackward(const XLATensor& grad_output, const XLATensor& input,
                           at::Scalar beta, at::Scalar threshold,
                           const XLATensor& output);

XLATensor Select(const XLATensor& input, xla::int64 dim, xla::int64 index);

XLATensor EmbeddingDenseBackward(const XLATensor& grad_output,
                                 const XLATensor& indices,
                                 xla::int64 num_weights, xla::int64 padding_idx,
                                 bool scale_grad_by_freq);

}  // namespace tensor_ops
}  // namespace swift_xla
