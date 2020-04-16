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

#include <unordered_map>
#include <vector>

#include "absl/types/span.h"
#include "absl/container/flat_hash_map.h"
#include "xla_tensor/ir.h"

namespace swift_xla {
namespace ir {

class Util {
 public:
  // Tracks the emission status of the nodes during the post-order generation.
  // It helps tracking loops within the computation graphs.
  enum EmitStatus {
    kNotEmitted,
    kEmitting,
    kEmitted,
  };

  using EmissionMap = absl::flat_hash_map<const Node*, EmitStatus>;

  // Computes the post order from the given node, without using recursion. The
  // emission map can be used as saved state, for multiple separate calls to
  // this API. The returned post-order can be empty if the node has already been
  // emitted inside the emission map. An error is generated if a loop is
  // detected.
  static std::vector<const Node*> ComputePostOrder(const Node* node,
                                                   EmissionMap* emap);

  static std::vector<const Node*> ComputePostOrder(
      absl::Span<const Node* const> nodes, EmissionMap* emap);

  // Same as above, but computes the post order on the set of nodes specified as
  // argument.
  static std::vector<const Node*> ComputePostOrder(
      absl::Span<const Node* const> nodes);

  // Clones the IR graph whose roots are passed in the values parameter.
  static std::vector<Value> Clone(absl::Span<const Value> values);

  // Same as the above, but the post-order is passed as parameter.
  static std::vector<Value> Clone(absl::Span<const Value> values,
                                  absl::Span<const Node* const> post_order);

  // Retrieves the number of nodes within the graph whose sink are passed in the
  // nodes argument.
  static size_t GetGraphSize(absl::Span<const Node* const> nodes);
};

}  // namespace ir
}  // namespace swift_xla
