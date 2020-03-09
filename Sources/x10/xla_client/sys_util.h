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

#ifndef X10_XLA_CLIENT_SYS_UTIL_H_
#define X10_XLA_CLIENT_SYS_UTIL_H_

#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace sys_util {

std::string GetEnvString(const char* name, const std::string& defval);

std::string GetEnvOrdinalPath(
    const char* name, const std::string& defval,
    const char* ordinal_env = "XRT_SHARD_LOCAL_ORDINAL");

int64 GetEnvInt(const char* name, int64 defval);

double GetEnvDouble(const char* name, double defval);

bool GetEnvBool(const char* name, bool defval);

// Retrieves the current EPOCH time in nanoseconds.
int64 NowNs();

}  // namespace sys_util
}  // namespace xla

#endif  // X10_XLA_CLIENT_SYS_UTIL_H_
