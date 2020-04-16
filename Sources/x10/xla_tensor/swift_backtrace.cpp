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

#include "tensorflow/compiler/tf2xla/xla_tensor/swift_backtrace.h"

#ifdef __linux__
#include <unistd.h>

#include <climits>

#include "absl/base/call_once.h"
#include "absl/debugging/stacktrace.h"
#include "absl/debugging/symbolize.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#endif

namespace swift_xla {

#ifdef __linux__
namespace {

absl::once_flag g_symbolizer_init_once;

void InitializeSymbolizer() {
  char self[PATH_MAX] = {0};
  int len = readlink("/proc/self/exe", self, sizeof(self));
  XLA_CHECK_GT(len, 0);
  absl::InitializeSymbolizer(self);
}

}  // namespace

std::vector<SourceLocation> GetSwiftFrames() {
  absl::call_once(g_symbolizer_init_once, InitializeSymbolizer);
  std::vector<SourceLocation> frames;
  int max_depth = 256;
  std::vector<void*> func_addr(max_depth);
  char func_name[1024];
  int depth = absl::GetStackTrace(func_addr.data(), max_depth, 0);
  for (int i = 0; i < depth; ++i) {
    bool success = absl::Symbolize(func_addr[i], func_name, sizeof(func_name));
    SourceLocation location;
    if (success) {
      location.function = func_name;
    } else {
      location.function = "(unknown)";
    }
    frames.push_back(location);
  }
  return frames;
}
#else
std::vector<SourceLocation> GetSwiftFrames() {
  std::vector<SourceLocation> frames;
  SourceLocation location;
  location.function = "(unknown)";
  frames.push_back(location);
  return frames;
}
#endif

std::ostream& operator<<(std::ostream& stream,
                         const std::vector<SourceLocation>& frames) {
  stream << "Swift Frames:\n";
  for (auto& location : frames) {
    stream << "  " << location.function << "\n";
  }
  return stream;
}

}  // namespace swift_xla
