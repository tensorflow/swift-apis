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

// Indexing tensors by by tensors
//
// This corresponds to "advanced indexing" in NumPy. The two operations are:
//
//  index(Tensor self, indices) -> Tensor
//  index_put_(Tensor self, indices, value, accumulate=false)
//
// The index is a TensorList containg kLong or kByte tensors or nulls. Byte
// tensors (boolean masks) are expanded to long tensors via nonzero(). Null
// tensors signify that the dimension is not indexed.
//
// All indexes are broadcast together and iterated as *one*. From NumPy:
//
// result[i_1, ..., i_M] == x[ind_1[i_1, ..., i_M], ind_2[i_1, ..., i_M],
//                           ..., ind_N[i_1, ..., i_M]]
//
// Note 1: ByteTensors expand to index as many dimensions as there are in the
// mask.
//
// Note 2: The behavior is more complicated when the index tensors are not all
// adjacent (e.g. x[[0, 1], :, [2, 3]]). In this case, self and the index
// tensors are transposed to the front: x.transpose(1, 2)[[0, 1], [2, 3]]

#pragma once

#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/tensor.h"

namespace swift_xla {

// Expands a rank <= 1 tensor to rank 1, if necessary.
ir::Value EnsureRank1(const ir::Value& index);

// Implements indexing by tensors of long according to the top-level
// description.
XLATensor IndexByTensors(const XLATensor& base,
                         absl::Span<const XLATensor> indices,
                         xla::int64 start_dim);

ir::Value IndexPutByTensors(const XLATensor& base,
                            absl::Span<const XLATensor> indices,
                            xla::int64 start_dim, const XLATensor& values,
                            bool accumulate,
                            absl::Span<const xla::int64> result_permutation);

ir::NodePtr IndexFill(const XLATensor& base, xla::int64 dim,
                      const XLATensor& index, at::Scalar value);

ir::NodePtr IndexFill(const XLATensor& base, xla::int64 dim,
                      const XLATensor& index, const XLATensor& value);

ir::Value IndexAdd(const XLATensor& base, xla::int64 dim,
                   const XLATensor& index, const XLATensor& source);

ir::Value IndexCopy(const XLATensor& base, xla::int64 dim,
                    const XLATensor& index, const XLATensor& source);

}  // namespace swift_xla
