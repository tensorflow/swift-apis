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

#include "xla_tensor/ops/infer_output_shape.h"

#include "xla_tensor/helpers.h"

namespace swift_xla {
namespace ir {
namespace ops {

xla::Shape InferOutputShape(absl::Span<const xla::Shape> input_shapes,
                            const LowerForShapeFn& core_lowering_fn) {
  xla::XlaBuilder b("InferOutputShape");
  std::vector<xla::XlaOp> parameters;
  for (size_t parameter_number = 0; parameter_number < input_shapes.size();
       ++parameter_number) {
    parameters.push_back(xla::Parameter(&b, parameter_number,
                                        input_shapes[parameter_number],
                                        absl::StrCat("p", parameter_number)));
  }
  xla::XlaOp result = core_lowering_fn(parameters);
  return XlaHelpers::ShapeOfXlaOp(result);
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
