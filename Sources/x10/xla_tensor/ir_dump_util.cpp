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

#include "tensorflow/compiler/tf2xla/xla_tensor/ir_dump_util.h"

#include <regex>
#include <sstream>
#include <unordered_map>

#include "absl/container/node_hash_map.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/debug_util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ir_util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/xla_util.h"

namespace swift_xla {
namespace ir {
namespace {

using NodeIdMap = absl::node_hash_map<const Node*, size_t>;

struct AttrTag {
  std::string name;
  std::string value;
  std::string::size_type pos;
};

std::string::size_type SkipTagSeparator(const std::string& node_string,
                                        std::string::size_type pos) {
  return node_string.compare(pos, 2, ", ") == 0 ? pos + 2 : pos;
}

absl::optional<AttrTag> ParseAttrTag(const std::string& node_string,
                                     std::string::size_type pos) {
  const std::regex tag_regex("^([a-zA-Z0-9_]+)=");
  std::smatch match;
  if (!std::regex_search(node_string.begin() + pos, node_string.end(), match,
                         tag_regex)) {
    return absl::nullopt;
  }

  std::string::size_type vpos = match[1].second - node_string.begin() + 1;
  int nested_open = -1;
  int nested_close = -1;
  size_t nest_count = 1;
  AttrTag tag;
  tag.name = match[1].str();
  for (pos = vpos; pos < node_string.size(); ++pos) {
    if (nested_open < 0) {
      if (SkipTagSeparator(node_string, pos) != pos) {
        break;
      }
      switch (node_string[pos]) {
        case '(':
          nested_open = node_string[pos];
          nested_close = ')';
          break;
        case '[':
          nested_open = node_string[pos];
          nested_close = ']';
          break;
        case '{':
          nested_open = node_string[pos];
          nested_close = '}';
          break;
      }
    } else if (node_string[pos] == nested_close) {
      --nest_count;
      if (nest_count == 0) {
        nest_count = 1;
        nested_open = nested_close = -1;
      }
    } else if (node_string[pos] == nested_open) {
      ++nest_count;
    }
  }
  tag.value = node_string.substr(vpos, pos - vpos);
  tag.pos = pos;
  return tag;
}

NodeIdMap GenerateIdMap(absl::Span<const Node* const> post_order) {
  NodeIdMap id_map;
  for (auto node : post_order) {
    XLA_CHECK(id_map.emplace(node, id_map.size()).second) << node->ToString();
  }
  return id_map;
}

std::unordered_map<const Node*, size_t> GetRootsIds(
    absl::Span<const Node* const> roots) {
  std::unordered_map<const Node*, size_t> roots_ids;
  for (size_t i = 0; i < roots.size(); ++i) {
    roots_ids[roots[i]] = i;
  }
  return roots_ids;
}

absl::optional<size_t> GetRootNodeId(
    const Node* node,
    const std::unordered_map<const Node*, size_t>& roots_ids) {
  auto it = roots_ids.find(node);
  if (it == roots_ids.end()) {
    return absl::nullopt;
  }
  return it->second;
}

std::vector<AttrTag> GetNodeTags(const Node* node) {
  std::string node_string = node->ToString();
  std::string op_string = node->op().ToString();
  std::string::size_type pos = node_string.find(op_string);
  XLA_CHECK_NE(pos, std::string::npos) << node_string << " : " << op_string;
  pos += op_string.size();
  std::vector<AttrTag> tags;
  for (;;) {
    pos = SkipTagSeparator(node_string, pos);
    auto tag = ParseAttrTag(node_string, pos);
    if (!tag) {
      break;
    }
    pos = tag->pos;
    tags.push_back(std::move(*tag));
  }
  return tags;
}

std::string GenerateDotNodeLabel(
    const Node* node,
    const std::unordered_map<const Node*, size_t>& roots_ids) {
  static const size_t kMaxValueSize = 64;
  std::stringstream ss;
  ss << node->op() << "\\n" << node->shape();
  for (auto& tag : GetNodeTags(node)) {
    ss << "\\n" << tag.name << "=";
    if (tag.value.size() < kMaxValueSize) {
      ss << tag.value;
    } else {
      ss << tag.value.substr(0, kMaxValueSize) << "...";
    }
  }
  auto opt_root_id = GetRootNodeId(node, roots_ids);
  if (opt_root_id) {
    ss << "\\nROOT=" << *opt_root_id;
  }
  return ss.str();
}

std::string GenerateDotNodeSpec(
    const Node* node,
    const std::unordered_map<const Node*, size_t>& roots_ids) {
  std::stringstream ss;
  ss << "label=\"" << GenerateDotNodeLabel(node, roots_ids) << "\"";
  return ss.str();
}

std::string GenerateTextNodeSpec(const Node* node, const NodeIdMap& id_map,
                                 bool with_shape) {
  std::stringstream ss;
  if (with_shape) {
    ss << node->shape() << " ";
  }
  ss << node->op() << "(";
  size_t count = 0;
  for (auto& output : node->operands()) {
    if (count > 0) {
      ss << ", ";
    }
    ss << "%" << id_map.at(output.node);
    if (output.node->num_outputs() > 1) {
      ss << "." << output.index;
    }
    ++count;
  }
  ss << ")";
  for (auto& tag : GetNodeTags(node)) {
    ss << ", " << tag.name << "=" << tag.value;
  }
  return ss.str();
}

std::string GenerateTextNodeSpec(const Node* node, const NodeIdMap& id_map) {
  return GenerateTextNodeSpec(node, id_map, true);
}

struct ChangeLogNode {
  std::string text;
  xla::Shape shape;
  std::vector<SourceLocation> backtrace;
};

thread_local std::map<xla::hash_t, std::vector<ChangeLogNode>> g_change_logs;

}  // namespace

std::string DumpUtil::ToDot(absl::Span<const Node* const> nodes) {
  auto post_order = Util::ComputePostOrder(nodes);
  return PostOrderToDot(post_order, nodes);
}

std::string DumpUtil::PostOrderToDot(absl::Span<const Node* const> post_order,
                                     absl::Span<const Node* const> roots) {
  std::unordered_map<const Node*, size_t> roots_ids = GetRootsIds(roots);
  NodeIdMap id_map = GenerateIdMap(post_order);
  std::stringstream ss;
  ss << "digraph G {\n";
  for (auto node : post_order) {
    ss << "  node" << id_map.at(node) << " ["
       << GenerateDotNodeSpec(node, roots_ids) << "]\n";
  }
  for (auto it = post_order.rbegin(); it != post_order.rend(); ++it) {
    const Node* node = *it;
    size_t id = id_map.at(node);
    for (size_t i = 0; i < node->operands().size(); ++i) {
      const ir::Output& output = node->operand(i);
      ss << "  node" << id_map.at(output.node) << " -> node" << id;
      if (node->operands().size() > 1) {
        ss << " [label=\"i=" << i;
        if (output.node->num_outputs() > 1) {
          ss << ",o=" << output.index;
        }
        ss << "\"]\n";
      } else {
        if (output.node->num_outputs() > 1) {
          ss << " [label=\"o=" << output.index << "\"]";
        }
        ss << "\n";
      }
    }
  }
  ss << "}\n";
  return ss.str();
}

std::string DumpUtil::ToText(absl::Span<const Node* const> nodes) {
  auto post_order = Util::ComputePostOrder(nodes);
  return PostOrderToText(post_order, nodes);
}

std::string DumpUtil::PostOrderToText(absl::Span<const Node* const> post_order,
                                      absl::Span<const Node* const> roots) {
  std::unordered_map<const Node*, size_t> roots_ids = GetRootsIds(roots);
  NodeIdMap id_map = GenerateIdMap(post_order);
  std::stringstream ss;
  ss << "IR {\n";
  for (auto node : post_order) {
    auto opt_root_id = GetRootNodeId(node, roots_ids);
    ss << "  %" << id_map.at(node) << " = "
       << GenerateTextNodeSpec(node, id_map);
    if (opt_root_id) {
      ss << ", ROOT=" << *opt_root_id;
    }
    ss << "\n";
  }
  ss << "}\n";
  return ss.str();
}

std::string DumpUtil::ToHlo(absl::Span<const Value> values,
                            const Device& device) {
  ir::LoweringContext lowering_ctx("IrToHlo", device);
  for (auto& ir_value : values) {
    xla::XlaOp root = lowering_ctx.GetOutputOp(ir_value);
    lowering_ctx.AddResult(root);
  }
  xla::XlaComputation computation = ConsumeValue(lowering_ctx.Build());
  return ConsumeValue(xla::util::GetComputationHloText(computation));
}

std::string DumpUtil::GetGraphChangeLog(absl::Span<const Node* const> roots) {
  auto post_order = Util::ComputePostOrder(roots);
  std::unordered_map<const Node*, size_t> roots_ids = GetRootsIds(roots);
  NodeIdMap id_map = GenerateIdMap(post_order);
  std::vector<ChangeLogNode> change_log;
  xla::hash_t h = 0x85ebca77c2b2ae63;
  for (auto node : post_order) {
    auto opt_root_id = GetRootNodeId(node, roots_ids);
    ChangeLogNode change_log_node;
    std::stringstream node_serializer;
    node_serializer << "%" << id_map.at(node) << " = "
                    << GenerateTextNodeSpec(node, id_map, false);
    if (opt_root_id) {
      node_serializer << ", ROOT=" << *opt_root_id;
    }
    change_log_node.text = node_serializer.str();
    h = xla::util::HashCombine(h, xla::util::Hash(change_log_node.text));
    change_log_node.shape = node->shape();
    change_log_node.backtrace = node->metadata().frame_info;
    change_log.push_back(change_log_node);
  }
  std::stringstream ss;
  const auto old_it = g_change_logs.find(h);
  if (old_it == g_change_logs.end()) {
    ss << "New graph structure";
  } else {
    const auto& old_change_log = old_it->second;
    for (size_t i = 0; i < post_order.size(); ++i) {
      const auto crt = change_log[i];
      const auto old = old_change_log[i];
      if (crt.shape != old.shape) {
        ss << "Found different shape: " << crt.shape << " vs " << old.shape
           << " for:\n  " << crt.text << "\n";
        ss << "Current trace:\n" << crt.backtrace;
        ss << "Previous trace:\n" << old.backtrace;
        break;
      }
    }
  }
  g_change_logs.emplace(h, change_log);
  return ss.str();
}

}  // namespace ir
}  // namespace swift_xla
