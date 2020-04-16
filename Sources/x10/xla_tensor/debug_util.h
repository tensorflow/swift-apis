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

#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "xla_tensor/tensor.h"

namespace swift_xla {

class DebugUtil {
 public:
  enum GraphFormat {
    kText,
    kDot,
    kHlo,
  };

  static GraphFormat GetDefaultGraphFormat();

  // Dumps the current Python frame and the IR Graph whose roots are the IR
  // values held at the tensors. If indices is not nullptr, it selects the
  // indices of the tensors whose graph will be emitted.
  static std::string GetTensorsGraphInfo(
      absl::Span<const XLATensor> tensors, const std::vector<size_t>* indices,
      GraphFormat format = GetDefaultGraphFormat());

  // If the environment variable XLA_SAVE_TENSORS_FILE is set to the proper
  // output path, an instance of the report returned by GetTensorsGraphInfo() is
  // saved.
  static void SaveTensorsGraphInfo(
      const char* name, absl::Span<const XLATensor> tensors,
      const std::vector<size_t>* indices,
      GraphFormat format = GetDefaultGraphFormat());

  static bool ExperimentEnabled(const std::string& name);
};

}  // namespace swift_xla
