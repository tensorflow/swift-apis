#pragma once

#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"

namespace swift_xla {
namespace ir {

NodePtr operator+(const Value& node1, const Value& node2);
NodePtr operator-(const Value& node1, const Value& node2);
NodePtr operator*(const Value& node1, const Value& node2);
NodePtr operator/(const Value& node1, const Value& node2);

}  // namespace ir
}  // namespace swift_xla
