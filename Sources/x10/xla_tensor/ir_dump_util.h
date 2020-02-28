#pragma once

#include <string>

#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"

namespace swift_xla {
namespace ir {

class DumpUtil {
 public:
  static std::string ToDot(absl::Span<const Node* const> nodes);

  static std::string PostOrderToDot(absl::Span<const Node* const> post_order,
                                    absl::Span<const Node* const> roots);

  static std::string ToText(absl::Span<const Node* const> nodes);

  static std::string PostOrderToText(absl::Span<const Node* const> post_order,
                                     absl::Span<const Node* const> roots);

  static std::string ToHlo(absl::Span<const Value> values);
};

}  // namespace ir
}  // namespace swift_xla
