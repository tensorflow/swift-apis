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

#ifndef X10_XLA_CLIENT_DEBUG_MACROS_H_
#define X10_XLA_CLIENT_DEBUG_MACROS_H_

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_client/tf_logging.h"
#include "tensorflow/core/platform/stacktrace.h"

#define XLA_ERROR() TF_ERROR_STREAM()
#define XLA_CHECK(c) TF_CHECK(c) << "\n" << tensorflow::CurrentStackTrace()
#define XLA_CHECK_OK(c) \
  TF_CHECK_OK(c) << "\n" << tensorflow::CurrentStackTrace()
#define XLA_CHECK_EQ(a, b) \
  TF_CHECK_EQ(a, b) << "\n" << tensorflow::CurrentStackTrace()
#define XLA_CHECK_NE(a, b) \
  TF_CHECK_NE(a, b) << "\n" << tensorflow::CurrentStackTrace()
#define XLA_CHECK_LE(a, b) \
  TF_CHECK_LE(a, b) << "\n" << tensorflow::CurrentStackTrace()
#define XLA_CHECK_GE(a, b) \
  TF_CHECK_GE(a, b) << "\n" << tensorflow::CurrentStackTrace()
#define XLA_CHECK_LT(a, b) \
  TF_CHECK_LT(a, b) << "\n" << tensorflow::CurrentStackTrace()
#define XLA_CHECK_GT(a, b) \
  TF_CHECK_GT(a, b) << "\n" << tensorflow::CurrentStackTrace()

template <typename T>
T ConsumeValue(xla::StatusOr<T>&& status) {
  XLA_CHECK_OK(status.status());
  return status.ConsumeValueOrDie();
}

#endif  // X10_XLA_CLIENT_DEBUG_MACROS_H_
