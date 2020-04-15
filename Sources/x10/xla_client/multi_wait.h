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

#ifndef X10_XLA_CLIENT_MULTI_WAIT_H_
#define X10_XLA_CLIENT_MULTI_WAIT_H_

#include <condition_variable>
#include <functional>
#include <mutex>

#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace util {

// Support waiting for a number of tasks to complete.
class MultiWait {
 public:
  explicit MultiWait(size_t count) : count_(count) {}

  // Signal the completion of a single task.
  void Done();

  // Waits until at least count (passed as constructor value) completions
  // happened.
  void Wait();

  // Same as above, but waits up to wait_seconds.
  void Wait(double wait_seconds);

  // Resets the threshold counter for the MultiWait object. The completed count
  // is also reset to zero.
  void Reset(size_t count);

  // Creates a completer functor which signals the mult wait object once func
  // has completed. Handles exceptions by signaling the multi wait with the
  // proper status value.
  std::function<void()> Completer(std::function<void()> func);

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  size_t count_ = 0;
  size_t completed_count_ = 0;
  std::exception_ptr exptr_;
};

}  // namespace util
}  // namespace xla

#endif  // X10_XLA_CLIENT_MULTI_WAIT_H_
