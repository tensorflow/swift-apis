#pragma once

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/tf_bit_generator.h"

namespace swift_xla {
namespace ir {
namespace ops {

class TfStatelessRandomUniform : public Node {
 public:
  TfStatelessRandomUniform(xla::Shape shape, const Value& seeds,
                           const Value& minval, const Value& maxval,
                           BitGeneratorType generator);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  BitGeneratorType generator() const { return generator_; }

 private:
  BitGeneratorType generator_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
