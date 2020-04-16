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

#include "xla_tensor/ops/tf_bit_generator.h"

namespace swift_xla {
namespace ir {
namespace ops {

xla::BitGeneratorTy GetBitGenerator(BitGeneratorType type) {
  switch (type) {
    case BitGeneratorType::PHILOX:
      return [](xla::XlaOp key, xla::XlaOp state, const xla::Shape& shape) {
        std::tie(state, key) = xla::ScramblePhiloxKey(key);
        return xla::PhiloxBitGenerator(key, state, shape);
      };
    case BitGeneratorType::THREE_FRY:
      return xla::ThreeFryBitGenerator;
  }
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
