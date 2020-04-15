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

#include "tensorflow/compiler/xla/xla_client/sys_util.h"

#include <chrono>
#include <cstdlib>
#include <cstring>

#include "absl/strings/str_cat.h"

namespace xla {
namespace sys_util {

std::string GetEnvString(const char* name, const std::string& defval) {
  const char* env = std::getenv(name);
  return env != nullptr ? env : defval;
}

std::string GetEnvOrdinalPath(const char* name, const std::string& defval,
                              const char* ordinal_env) {
  std::string path = GetEnvString(name, defval);
  if (!path.empty()) {
    int64 ordinal = GetEnvInt(ordinal_env, -1);
    if (ordinal >= 0) {
      absl::StrAppend(&path, ".", ordinal);
    }
  }
  return path;
}

int64 GetEnvInt(const char* name, int64 defval) {
  const char* env = std::getenv(name);
  return env != nullptr ? std::atol(env) : defval;
}

double GetEnvDouble(const char* name, double defval) {
  const char* env = std::getenv(name);
  return env != nullptr ? std::atof(env) : defval;
}

bool GetEnvBool(const char* name, bool defval) {
  const char* env = std::getenv(name);
  if (env == nullptr) {
    return defval;
  }
  if (std::strcmp(env, "true") == 0) {
    return true;
  }
  if (std::strcmp(env, "false") == 0) {
    return false;
  }
  return std::atoi(env) != 0;
}

int64 NowNs() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             now.time_since_epoch())
      .count();
}

}  // namespace sys_util
}  // namespace xla
