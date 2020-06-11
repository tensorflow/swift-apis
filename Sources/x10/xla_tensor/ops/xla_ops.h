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

#include <string>

#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"

namespace swift_xla {
namespace ir {
namespace ops {

class OpKindWrapper {
 public:
  explicit OpKindWrapper(c10::Symbol symbol) : op_kind_(symbol) {}

  const OpKind& operator*() const { return get(); }

  operator OpKind() const { return get(); }

 private:
  const OpKind& get() const { return op_kind_; }
  OpKind op_kind_;
};

extern const OpKindWrapper xla_all_to_all;
extern const OpKindWrapper xla_as_strided_view_update;
extern const OpKindWrapper xla_cast;
extern const OpKindWrapper xla_collective_permute;
extern const OpKindWrapper xla_cross_replica_sum;
extern const OpKindWrapper xla_device_data;
extern const OpKindWrapper xla_diagonal_view_update;
extern const OpKindWrapper xla_generic_slice;
extern const OpKindWrapper xla_get_dimensions_size;
extern const OpKindWrapper xla_moving_average;
extern const OpKindWrapper xla_nms;
extern const OpKindWrapper xla_not_supported;
extern const OpKindWrapper xla_replication_pad;
extern const OpKindWrapper xla_replication_pad_backward;
extern const OpKindWrapper xla_select;
extern const OpKindWrapper xla_tensor_data;
extern const OpKindWrapper xla_token;
extern const OpKindWrapper xla_unselect;
extern const OpKindWrapper xla_update_slice;

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
