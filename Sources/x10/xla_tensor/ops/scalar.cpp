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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/scalar.h"

#include <functional>
#include <sstream>

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/aten_compat.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace swift_xla {
namespace ir {
namespace ops {

Scalar::Scalar(at::Scalar value, xla::Shape shape)
    : Node(OpKind(at::prim::Constant), std::move(shape), /*num_outputs=*/1,
           ScalarHash(value)),
      value_(std::move(value)) {}

Scalar::Scalar(at::Scalar value, xla::PrimitiveType type)
    : Node(OpKind(at::prim::Constant), xla::ShapeUtil::MakeShape(type, {}),
           /*num_outputs=*/1, ScalarHash(value)),
      value_(std::move(value)) {}

std::string Scalar::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", value=" << value_;
  return ss.str();
}

NodePtr Scalar::Clone(OpList operands) const {
  return MakeNode<Scalar>(value_, shape());
}

XlaOpVector Scalar::Lower(LoweringContext* loctx) const {
  xla::Literal literal(xla::ShapeUtil::MakeShape(shape().element_type(), {}));
  switch (shape().element_type()) {
    case xla::PrimitiveType::PRED:
      literal.Set<bool>({}, static_cast<bool>(value_.toInt()));
      break;
    case xla::PrimitiveType::S8:
      literal.Set<xla::int8>({}, static_cast<xla::int8>(value_.toChar()));
      break;
    case xla::PrimitiveType::U8:
      literal.Set<xla::uint8>({}, static_cast<xla::uint8>(value_.toByte()));
      break;
    case xla::PrimitiveType::S16:
      literal.Set<xla::int16>({}, static_cast<xla::int16>(value_.toShort()));
      break;
    case xla::PrimitiveType::U16:
      literal.Set<xla::uint16>({}, static_cast<xla::uint16>(value_.toShort()));
      break;
    case xla::PrimitiveType::S32:
      literal.Set<xla::int32>({}, static_cast<xla::int32>(value_.toInt()));
      break;
    case xla::PrimitiveType::U32:
      literal.Set<xla::uint32>({}, static_cast<xla::uint32>(value_.toInt()));
      break;
    case xla::PrimitiveType::S64:
      literal.Set<xla::int64>({}, static_cast<xla::int64>(value_.toLong()));
      break;
    case xla::PrimitiveType::U64:
      literal.Set<xla::uint64>({}, static_cast<xla::uint64>(value_.toLong()));
      break;
    case xla::PrimitiveType::F32:
      literal.Set<float>({}, static_cast<float>(value_.toDouble()));
      break;
    case xla::PrimitiveType::F64:
      literal.Set<double>({}, value_.toDouble());
      break;
    case xla::PrimitiveType::BF16:
      literal.Set<xla::bfloat16>({},
                                 static_cast<xla::bfloat16>(value_.toDouble()));
      break;
    case xla::PrimitiveType::F16:
      literal.Set<xla::half>({}, static_cast<xla::half>(value_.toDouble()));
      break;
    default: {
      std::stringstream ss;
      ss << value_;
      XLA_ERROR() << "Unable to lower scalar " << ss.str() << " of shape "
                  << shape();
    }
  }

  xla::XlaOp op = xla::ConstantLiteral(loctx->builder(), literal);
  if (shape().rank() > 0) {
    op = xla::Broadcast(op, shape().dimensions());
  }
  return ReturnOp(op, loctx);
}

xla::hash_t ScalarHash(at::Scalar s) {
  return s.isFloatingPoint() ? xla::util::Hash(s.toDouble())
                             : xla::util::Hash(s.toLong());
}

std::ostream& operator<<(std::ostream& ostrm, at::Scalar s) {
  return ostrm << (s.isFloatingPoint() ? s.toDouble() : s.toLong());
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
