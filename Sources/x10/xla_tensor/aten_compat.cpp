#include "tensorflow/compiler/tf2xla/xla_tensor/aten_compat.h"

namespace c10 {

const char* Symbol::toQualString() const {
  switch (value) {
#define HANDLE_KEY(ns, s) \
  case at::aten::s:       \
    return "aten::" #s;
    FORALL_ATEN_BASE_SYMBOLS(HANDLE_KEY)
#undef HANDLE_KEY
#define HANDLE_KEY(ns, s)         \
  case swift_xla::xla_symbols::s: \
    return "xla::" #s;
    FORALL_XLA_SYMBOLS(HANDLE_KEY, HANDLE_KEY)
#undef HANDLE_KEY
    case at::prim::Constant:
      return "prim::Constant";
    default:
      return "<?>";
  }
}

}  // namespace c10
