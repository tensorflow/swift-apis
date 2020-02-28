#pragma once

#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"

namespace swift_xla {
namespace ir {
namespace ops {

// Value has implicit cast to bool, operator overloads would be confusing.
Value BitwiseAnd(const Value& node1, const Value& node2);
Value BitwiseOr(const Value& node1, const Value& node2);
Value BitwiseXor(const Value& node1, const Value& node2);

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
