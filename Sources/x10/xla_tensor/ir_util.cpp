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

#include "tensorflow/compiler/tf2xla/xla_tensor/ir_util.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace swift_xla {
namespace ir {

std::vector<const Node*> Util::ComputePostOrder(const Node* node,
                                                EmissionMap* emap) {
  std::vector<const Node*> post_order;
  std::vector<const Node*> queue;
  queue.push_back(node);
  while (!queue.empty()) {
    node = queue.back();
    auto emplace_result = emap->emplace(node, kEmitting);
    auto it = emplace_result.first;
    if (emplace_result.second) {
      for (const auto& output : node->operands()) {
        auto oit = emap->find(output.node);
        if (oit == emap->end()) {
          queue.push_back(output.node);
        } else if (oit->second == kEmitting) {
          XLA_ERROR() << "Graph loop found at " << *output.node;
        }
      }
      if (node->operands().empty()) {
        post_order.push_back(node);
        it->second = kEmitted;
        queue.pop_back();
      }
    } else if (it->second == kEmitting) {
#ifndef NDEBUG
      for (auto& output : node->operands()) {
        auto oit = emap->find(output.node);
        XLA_CHECK(oit != emap->end() && oit->second == kEmitted)
            << "Graph loop found at " << *output.node;
      }
#endif
      it->second = kEmitted;
      post_order.push_back(node);
      queue.pop_back();
    } else {
      XLA_CHECK_EQ(it->second, kEmitted);
      queue.pop_back();
    }
  }
  return post_order;
}

std::vector<const Node*> Util::ComputePostOrder(
    absl::Span<const Node* const> nodes, EmissionMap* emap) {
  std::vector<const Node*> post_order;
  for (auto node : nodes) {
    auto node_post_order = ComputePostOrder(node, emap);
    post_order.insert(post_order.end(), node_post_order.begin(),
                      node_post_order.end());
  }
  return post_order;
}

std::vector<const Node*> Util::ComputePostOrder(
    absl::Span<const Node* const> nodes) {
  EmissionMap emap;
  return ComputePostOrder(nodes, &emap);
}

std::vector<Value> Util::Clone(absl::Span<const Value> values,
                               absl::Span<const Node* const> post_order) {
  std::unordered_map<const Node*, NodePtr> clone_map;
  for (auto node : post_order) {
    if (clone_map.count(node) > 0) {
      continue;
    }
    std::vector<Value> inputs;
    for (auto& output : node->operands()) {
      auto it = clone_map.find(output.node);
      XLA_CHECK(it != clone_map.end())
          << "Bad post-order: " << node->ToString();
      inputs.emplace_back(it->second, output.index);
    }
    clone_map[node] = node->Clone(inputs);
  }

  std::vector<Value> cloned;
  for (auto& value : values) {
    auto it = clone_map.find(value.node.get());
    XLA_CHECK(it != clone_map.end()) << "Bad post-order: " << value->ToString();
    cloned.emplace_back(it->second, value.index);
  }
  return cloned;
}

std::vector<Value> Util::Clone(absl::Span<const Value> values) {
  std::vector<const Node*> nodes;
  for (auto& value : values) {
    nodes.push_back(value.node.get());
  }
  std::vector<const Node*> post_order = ComputePostOrder(nodes);
  return Clone(values, post_order);
}

size_t Util::GetGraphSize(absl::Span<const Node* const> nodes) {
  std::vector<const Node*> post_order = ComputePostOrder(nodes);
  return post_order.size();
}

}  // namespace ir
}  // namespace swift_xla
