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

#ifndef X10_XLA_CLIENT_METRICS_READER_H_
#define X10_XLA_CLIENT_METRICS_READER_H_

#include <string>

namespace xla {
namespace metrics_reader {

// Creates a report with the current metrics statistics.
std::string CreateMetricReport();

}  // namespace metrics_reader
}  // namespace xla

#endif  // X10_XLA_CLIENT_METRICS_READER_H_
