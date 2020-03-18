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

#include "tensorflow/compiler/xla/xla_client/xrt_session.h"

#include "absl/strings/str_cat.h"

namespace xla {

XrtSession::XrtSession(const tensorflow::SessionOptions& session_options)
    : target_(session_options.target),
      root_(tensorflow::Scope::NewRootScope()),
      session_(root_, session_options) {}

void XrtSession::Reset() {
  for (auto& name_cache : node_cache_) {
    name_cache.second.Rewind();
  }
}

std::string XrtSession::GetCacheKey(const std::string& op_name,
                                    const std::string& device) {
  return absl::StrCat(op_name, ";", device);
}

}  // namespace xla
