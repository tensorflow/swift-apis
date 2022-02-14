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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/ops.h"

#include <cmath>

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/convert_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/data_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/elementwise.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/constant.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/tensor_util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/xla_lower_util.h"
#include "tensorflow/compiler/tf2xla/lib/random.h"
#include "tensorflow/compiler/xla/client/lib/logdet.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace swift_xla {
namespace ir {
namespace ops {

NodePtr ARange(at::Scalar start, at::Scalar end, at::Scalar step,
               at::ScalarType scalar_type) {
  xla::PrimitiveType type = MakeXlaPrimitiveType(scalar_type,
                                                 /*device=*/nullptr);
  XLA_CHECK_NE(step.toDouble(), 0.0);
  XLA_CHECK(!std::isnan(start.toDouble()) && !std::isnan(end.toDouble()))
      << "unsupported range: " << start.toDouble() << " -> " << end.toDouble();
  XLA_CHECK((start.toDouble() <= end.toDouble() && step.toDouble() > 0.0) ||
            (start.toDouble() >= end.toDouble() && step.toDouble() < 0.0));
  xla::Literal values;
  switch (type) {
    case xla::PrimitiveType::BF16:
      values = XlaHelpers::Range<tensorflow::bfloat16>(
          static_cast<tensorflow::bfloat16>(start.toFloat()),
          static_cast<tensorflow::bfloat16>(end.toFloat()),
          static_cast<tensorflow::bfloat16>(step.toFloat()));
      break;
    case xla::PrimitiveType::F32:
      values = XlaHelpers::Range<float>(start.toFloat(), end.toFloat(),
                                        step.toFloat());
      break;
    case xla::PrimitiveType::F64:
      values = XlaHelpers::Range<double>(start.toDouble(), end.toDouble(),
                                         step.toDouble());
      break;
    case xla::PrimitiveType::U8:
      values = XlaHelpers::Range<xla::uint8>(start.toByte(), end.toByte(),
                                             step.toByte());
      break;
    case xla::PrimitiveType::S8:
      values = XlaHelpers::Range<xla::int8>(start.toChar(), end.toChar(),
                                            step.toChar());
      break;
    case xla::PrimitiveType::S16:
      values = XlaHelpers::Range<xla::int16>(start.toShort(), end.toShort(),
                                             step.toShort());
      break;
    case xla::PrimitiveType::U16:
      values = XlaHelpers::Range<xla::uint16>(start.toInt(), end.toInt(),
                                              step.toInt());
      break;
    case xla::PrimitiveType::S32:
      values = XlaHelpers::Range<xla::int32>(start.toInt(), end.toInt(),
                                             step.toInt());
      break;
    case xla::PrimitiveType::U32:
      values = XlaHelpers::Range<xla::uint32>(start.toLong(), end.toLong(),
                                              step.toLong());
      break;
    case xla::PrimitiveType::S64:
      values = XlaHelpers::Range<int64_t>(start.toLong(), end.toLong(),
                                             step.toLong());
      break;
    case xla::PrimitiveType::U64:
      values = XlaHelpers::Range<xla::uint64>(start.toLong(), end.toLong(),
                                              step.toLong());
      break;
    default:
      XLA_ERROR() << "XLA type not supported: " << type;
  }
  return MakeNode<Constant>(std::move(values));
}

NodePtr LinSpace(at::Scalar start, at::Scalar stop, int64_t num,
                 at::ScalarType scalar_type) {
  XLA_CHECK_GT(num, 0) << "Requires num > 0: " << num;
  xla::PrimitiveType type = MakeXlaPrimitiveType(scalar_type,
                                                 /*device=*/nullptr);
  xla::Literal values;
  switch (type) {
    case xla::PrimitiveType::F32:
      values =
          XlaHelpers::LinSpace<float>(start.toFloat(), stop.toFloat(), num);
      break;
    case xla::PrimitiveType::F64:
      values =
          XlaHelpers::LinSpace<double>(start.toDouble(), stop.toDouble(), num);
      break;
    case xla::PrimitiveType::U8:
      values =
          XlaHelpers::LinSpace<xla::uint8>(start.toByte(), stop.toByte(), num);
      break;
    case xla::PrimitiveType::S8:
      values =
          XlaHelpers::LinSpace<xla::int8>(start.toChar(), stop.toChar(), num);
      break;
    case xla::PrimitiveType::S16:
      values = XlaHelpers::LinSpace<xla::int16>(start.toShort(), stop.toShort(),
                                                num);
      break;
    case xla::PrimitiveType::U16:
      values =
          XlaHelpers::LinSpace<xla::uint16>(start.toInt(), stop.toInt(), num);
      break;
    case xla::PrimitiveType::S32:
      values =
          XlaHelpers::LinSpace<xla::int32>(start.toInt(), stop.toInt(), num);
      break;
    case xla::PrimitiveType::U32:
      values =
          XlaHelpers::LinSpace<xla::uint32>(start.toLong(), stop.toLong(), num);
      break;
    case xla::PrimitiveType::S64:
      values =
          XlaHelpers::LinSpace<int64_t>(start.toLong(), stop.toLong(), num);
      break;
    case xla::PrimitiveType::U64:
      values =
          XlaHelpers::LinSpace<xla::uint64>(start.toLong(), stop.toLong(), num);
      break;
    default:
      XLA_ERROR() << "XLA type not supported: " << type;
  }
  return MakeNode<Constant>(std::move(values));
}

NodePtr BroadcastTensors(absl::Span<const Value> tensors) {
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    std::vector<xla::XlaOp> xla_operands;
    for (const Output& operand : node.operands()) {
      xla_operands.push_back(loctx->GetOutputOp(operand));
    }
    return node.ReturnOps(CreateBroadcastTensors(xla_operands), loctx);
  };
  std::vector<xla::Shape> tensor_shapes;
  for (const Value& tensor : tensors) {
    tensor_shapes.push_back(tensor.shape());
  }
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    auto results = CreateBroadcastTensors(operands);
    return xla::Tuple(results.front().builder(), results);
  };
  return GenericOp(
      OpKind(at::aten::broadcast_tensors), tensors,
      [&]() { return InferOutputShape(tensor_shapes, lower_for_shape_fn); },
      std::move(lower_fn), /*num_outputs=*/tensors.size());
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla

