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

#if defined(_WIN32)
#define XLA_API __declspec(dllexport)
#else
#define XLA_API __attribute__((__visibility__("default")))
#endif

#include "xla_tensor_wrapper.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"

using swift_xla::XLATensor;
using swift_xla::ir::LoweringContext;
using swift_xla::ir::Node;
using swift_xla::ir::NodePtr;
using swift_xla::ir::OpList;
using swift_xla::ir::Output;
using swift_xla::ir::Value;
using swift_xla::ir::XlaOpVector;

xla::Shape ShapeOfXlaOpList(absl::Span<const Value> ops) {
  xla::Shape result;
  result.set_element_type(xla::TUPLE);
  result.mutable_tuple_shapes()->reserve(ops.size());
  for (const auto& op : ops) {
    xla::ShapeUtil::AppendShapeToTuple(op.shape(), &result);
  }
  TF_DCHECK_OK(xla::ShapeUtil::ValidateShapeWithOptionalLayout(result));
  return result;
}

struct ExtraInputDiscovery {
  // TODO: color when building the graph as this can be n^2
  // in the number of for loops.
  void BackRefVisit(const Output& v, const Node* node = nullptr) {
    auto& state = state_map[v.node];
    if (!state.visited) {
      state.visited = true;
      work_list.push_back(v.node);
    }
    if (node) state.refs.push_back(node);
  }
  void PlaceholderVisit(const Node* node) {
    auto& state = state_map[node];
    if (!state.depends_on_placeholder) {
      state.depends_on_placeholder = true;
      work_list.push_back(node);
    }
  }
  void WorkListBackRefVisit() {
    while (!work_list.empty()) {
      const Node* node = work_list.back();
      work_list.pop_back();
      for (const auto& value : node->operands()) {
        BackRefVisit(value, node);
      }
    }
  }
  void WorkListPlaceholderVisit() {
    while (!work_list.empty()) {
      const Node* node = work_list.back();
      work_list.pop_back();
      for (auto* ref : state_map[node].refs) {
        PlaceholderVisit(ref);
      }
    }
  }
  void BackRefVisitExtraSearch(const Output& v, const NodePtr& n) {
    auto& state = state_map[v.node];
    if (!state.visited_looking_for_extras) {
      state.visited_looking_for_extras = true;
      if (state.depends_on_placeholder) {
        work_list.push_back(v.node);
      } else {
        results.push_back(Value(n, v.index));
      }
    }
  }
  void WorkListBackRefVisitExtraSearch() {
    while (!work_list.empty()) {
      const Node* node = work_list.back();
      work_list.pop_back();
      auto& operands = node->operands();
      auto& node_ptrs = node->operand_nodes();
      for (size_t i = 0; i < operands.size(); ++i) {
        BackRefVisitExtraSearch(operands[i], node_ptrs[i]);
      }
    }
  }
  struct State {
    State() {}
    bool visited =
        false;  // Has been fully visited if true and work_list.empty().
    bool depends_on_placeholder = false;
    bool visited_looking_for_extras = false;
    std::vector<const Node*> refs;
  };
  std::vector<const Node*> work_list;
  absl::flat_hash_map<const Node*, State> state_map;
  std::vector<Value> results;
};

std::vector<Value> DiscoverExtraInputs(absl::Span<const Value> results,
                                       const Value& index_placeholder,
                                       absl::Span<const Value> placeholders) {
  ExtraInputDiscovery state;
  for (auto& result : results) {
    state.BackRefVisit(result);
  }
  state.WorkListBackRefVisit();
  for (auto& placeholder : placeholders) {
    state.PlaceholderVisit(placeholder.node.get());
  }
  state.PlaceholderVisit(index_placeholder.node.get());
  state.WorkListPlaceholderVisit();
  for (auto& result : results) {
    state.BackRefVisitExtraSearch(result, result.node);
  }
  state.WorkListBackRefVisitExtraSearch();
  return std::move(state.results);
}

class XLAFunctionalWhileNode : public swift_xla::ir::Node {
 public:
  static std::vector<Value> BuildArgs(absl::Span<const Value> initial,
                                      const Value& n,
                                      absl::Span<const Value> extras) {
    std::vector<Value> out(initial.begin(), initial.end());
    out.push_back(n);
    out.insert(out.end(), extras.begin(), extras.end());
    return out;
  }
  static xla::hash_t HashOfResults(absl::Span<const Value> results) {
    xla::hash_t hash = 0;
    for (auto& result : results)
      hash = xla::util::HashCombine(hash, result.hash());
    return hash;
  }
  XLAFunctionalWhileNode(absl::Span<const Value> initial, const Value& n,
                         const Value& index_placeholder,
                         absl::Span<const Value> placeholders,
                         absl::Span<const Value> results)
      : Node(swift_xla::ir::OpKind(at::aten::functional_while),
             BuildArgs(
                 initial, n,
                 DiscoverExtraInputs(results, index_placeholder, placeholders)),
             ShapeOfXlaOpList(results), results.size(), HashOfResults(results)),
        index_placeholder_(index_placeholder),
        placeholders_(placeholders.begin(), placeholders.end()),
        results_(results.begin(), results.end()) {}

  static xla::XlaOp zeroLike(xla::XlaOp op) {
    auto* b = op.builder();
    return xla::ConstantLiteral(
        b, xla::LiteralUtil::Zero(
               swift_xla::XlaHelpers::ShapeOfXlaOp(op).element_type()));
  }

  static xla::XlaOp oneLike(xla::XlaOp op) {
    auto* b = op.builder();
    return xla::ConstantLiteral(
        b, xla::LiteralUtil::One(
               swift_xla::XlaHelpers::ShapeOfXlaOp(op).element_type()));
  }

  XlaOpVector Lower(LoweringContext* loctx) const {
    size_t last_i = placeholders_.size();

    auto body_builder = loctx->builder()->CreateSubBuilder("loop_body");
    xla::XlaOp initial;
    {
      std::vector<xla::XlaOp> args;
      args.reserve(operands().size() + 1);
      for (size_t i = 0; i < last_i; ++i) {
        args.push_back(loctx->GetOutputOp(operand(i)));
      }
      auto tmp = loctx->GetOutputOp(operand(last_i));
      auto it = zeroLike(tmp);
      args.push_back(it);
      args.push_back(tmp);
      for (size_t i = last_i + 1; i < operands().size(); ++i) {
        args.push_back(loctx->GetOutputOp(operand(i)));
      }

      initial = xla::Tuple(loctx->builder(), args);
    }
    xla::XlaOp body_result;
    {
      auto* b = body_builder.get();
      swift_xla::ir::Util::EmissionMap emap;
      for (const auto& placeholder : placeholders_) {
        emap[placeholder.node.get()] = swift_xla::ir::Util::kEmitted;
      }
      for (size_t i = last_i + 1; i < operands().size(); ++i) {
        emap[operand(i).node] = swift_xla::ir::Util::kEmitted;
      }
      emap[index_placeholder_.node.get()] = swift_xla::ir::Util::kEmitted;
      swift_xla::ir::LoweringContext body_loctx(b, loctx->device(),
                                                std::move(emap));
      auto t = xla::Parameter(
          b, 0, swift_xla::XlaHelpers::ShapeOfXlaOp(initial), "tuple");
      auto p1 = xla::GetTupleElement(t, last_i);
      auto p2 = xla::GetTupleElement(t, last_i + 1);
      for (size_t i = 0; i < placeholders_.size(); ++i) {
        body_loctx.AssignOutputOp(placeholders_[i], xla::GetTupleElement(t, i));
      }
      for (size_t i = last_i + 1; i < operands().size(); ++i) {
        body_loctx.AssignOutputOp(operand(i), xla::GetTupleElement(t, i + 1));
      }
      body_loctx.AssignOutputOp(index_placeholder_, p1);

      std::vector<xla::XlaOp> tmps;
      for (auto& result : results_) {
        tmps.push_back(body_loctx.GetOutputOp(result));
      }
      tmps.push_back(p1 + oneLike(p1));
      tmps.push_back(p2);
      for (size_t i = last_i + 1; i < operands().size(); ++i) {
        tmps.push_back(body_loctx.GetOutputOp(operand(i)));
      }
      body_result = xla::Tuple(b, tmps);
    }

    auto cond_builder = loctx->builder()->CreateSubBuilder("cond_body");
    xla::XlaOp cond_result;
    {
      auto* b = cond_builder.get();
      auto t = xla::Parameter(
          b, 0, swift_xla::XlaHelpers::ShapeOfXlaOp(initial), "tuple");
      auto p1 = xla::GetTupleElement(t, last_i);
      auto p2 = xla::GetTupleElement(t, last_i + 1);
      cond_result = xla::Lt(p1, p2);
    }

    auto result = xla::While(
        cond_builder->Build(cond_result).ConsumeValueOrDie(),
        body_builder->Build(body_result).ConsumeValueOrDie(), initial);

    std::vector<xla::XlaOp> results;
    for (size_t i = 0; i < last_i; ++i) {
      results.push_back(xla::GetTupleElement(result, i));
    }
    return ReturnOps(results, loctx);
  }

  Value index_placeholder_;
  std::vector<Value> placeholders_;
  std::vector<Value> results_;
};

class XLAPlaceholderNode : public swift_xla::ir::Node {
 public:
  XLAPlaceholderNode(xla::Shape shape, int id)
      : Node(swift_xla::ir::OpKind(at::aten::placeholder), {}, shape, 1,
             xla::util::MHash(id)),
        id_(id) {}
  NodePtr Clone(OpList operands) const override {
    return swift_xla::ir::MakeNode<XLAPlaceholderNode>(shape(), id_);
  }
  XlaOpVector Lower(LoweringContext* loctx) const override {
    LOG(FATAL) << "Cannot lower placeholder: " << ToString() << " id: " << id_;
  }
  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", id=" << id_;
    return ss.str();
  }
  int id_;
};

std::vector<Value> UnpackIrValues(OpaqueXLATensorArrayRef array) {
  std::vector<Value> out;
  out.reserve(array.size);
  for (size_t i = 0; i < array.size; ++i) {
    out.push_back(array.data[i]->GetIrValue());
  }
  return out;
}

OpaqueXLATensorArrayRef XLATensor_functional_while(
    OpaqueXLATensor* n, OpaqueXLATensorArrayRef initial,
    OpaqueXLATensorArrayRef placeholders, OpaqueXLATensor* indexPlaceholder,
    OpaqueXLATensorArrayRef results) {
  auto initial_ir = UnpackIrValues(initial);
  auto placeholders_ir = UnpackIrValues(placeholders);
  auto results_ir = UnpackIrValues(results);

  auto result_node = swift_xla::ir::MakeNode<XLAFunctionalWhileNode>(
      initial_ir, n->GetIrValue(), indexPlaceholder->GetIrValue(),
      placeholders_ir, results_ir);
  size_t count = results.size;
  auto opaque_tensors = new OpaqueXLATensor*[count];
  for (size_t i = 0; i < count; ++i) {
    opaque_tensors[i] = new XLATensor(
        results.data[i]->CreateFrom(swift_xla::ir::Value(result_node, i)));
  }
  return {opaque_tensors, count};
}

OpaqueXLATensor* XLATensor_makePlaceholder(OpaqueXLATensor* t, int id) {
  return new XLATensor(t->CreateFrom(
      swift_xla::ir::MakeNode<XLAPlaceholderNode>(t->shape(), id)));
}
