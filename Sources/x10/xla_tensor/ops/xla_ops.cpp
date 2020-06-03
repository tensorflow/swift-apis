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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/xla_ops.h"

#include "tensorflow/compiler/tf2xla/xla_tensor/aten_compat.h"

namespace swift_xla {
namespace ir {
namespace ops {

const OpKindWrapper xla_all_to_all(xla_symbols::all_to_all);
const OpKindWrapper xla_as_strided_view_update(
    xla_symbols::as_strided_view_update);
const OpKindWrapper xla_cast(xla_symbols::cast);
const OpKindWrapper xla_collective_permute(xla_symbols::collective_permute);
const OpKindWrapper xla_cross_replica_sum(xla_symbols::cross_replica_sum);
const OpKindWrapper xla_device_data(xla_symbols::device_data);
const OpKindWrapper xla_diagonal_view_update(xla_symbols::diagonal_view_update);
const OpKindWrapper xla_generic_slice(xla_symbols::generic_slice);
const OpKindWrapper xla_get_dimensions_size(xla_symbols::get_dimensions_size);
const OpKindWrapper xla_moving_average(xla_symbols::moving_average);
const OpKindWrapper xla_nms(xla_symbols::nms);
const OpKindWrapper xla_not_supported(xla_symbols::not_supported);
const OpKindWrapper xla_replication_pad(xla_symbols::replication_pad);
const OpKindWrapper xla_replication_pad_backward(
    xla_symbols::replication_pad_backward);
const OpKindWrapper xla_select(xla_symbols::select);
const OpKindWrapper xla_tensor_data(xla_symbols::tensor_data);
const OpKindWrapper xla_token(xla_symbols::token);
const OpKindWrapper xla_unselect(xla_symbols::unselect);
const OpKindWrapper xla_update_slice(xla_symbols::update_slice);

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
