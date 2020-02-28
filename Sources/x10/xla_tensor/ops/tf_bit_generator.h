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
