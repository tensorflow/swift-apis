#include "tensorflow/compiler/tf2xla/xla_tensor/ops/tf_bit_generator.h"

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
