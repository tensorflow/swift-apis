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

#include "xla_tensor/aten_compat.h"

namespace c10 {

const char* Symbol::toQualString() const {
  switch (value) {
#define HANDLE_KEY(ns, s) \
  case at::aten::s:       \
    return "x10::" #s;
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
