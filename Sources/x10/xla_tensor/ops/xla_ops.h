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

extern const OpKindWrapper xla_as_strided_view_update;
extern const OpKindWrapper xla_cast;
extern const OpKindWrapper xla_cross_replica_sum;
extern const OpKindWrapper xla_device_data;
extern const OpKindWrapper xla_diagonal_view_update;
extern const OpKindWrapper xla_generic_slice;
extern const OpKindWrapper xla_get_dimensions_size;
extern const OpKindWrapper xla_moving_average;
extern const OpKindWrapper xla_not_supported;
extern const OpKindWrapper xla_select;
extern const OpKindWrapper xla_tensor_data;
extern const OpKindWrapper xla_token;
extern const OpKindWrapper xla_unselect;
extern const OpKindWrapper xla_update_slice;

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
