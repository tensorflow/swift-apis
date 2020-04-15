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

#ifndef X10_XLA_TENSOR_OPS_TF_BIT_GENERATOR_H_
#define X10_XLA_TENSOR_OPS_TF_BIT_GENERATOR_H_

#include "tensorflow/compiler/xla/client/lib/prng.h"

namespace swift_xla {
namespace ir {
namespace ops {

enum class BitGeneratorType {
  PHILOX,
  THREE_FRY,
};

xla::BitGeneratorTy GetBitGenerator(BitGeneratorType type);

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla

#endif  // X10_XLA_TENSOR_OPS_TF_BIT_GENERATOR_H_
