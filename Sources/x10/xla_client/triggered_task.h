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

#ifndef X10_XLA_CLIENT_TRIGGERED_TASK_H_
#define X10_XLA_CLIENT_TRIGGERED_TASK_H_

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace xla {
namespace util {

// Wraps a function which should be run many times upon user activations.
class TriggeredTask {
 public:
  // Note that if num_threads > 1, the function will be run concurrently from
  // multiple threads, so it will have to be thread safe. This condition does
  // not apply if num_threads is 1.
  TriggeredTask(std::function<void()> function, size_t num_threads);

  // Stops the background thread and waits for it to complete.
  void Stop();

  // Triggers a function run. If the function is already running, it will run
  // again immediately after it completes. Returns tthe value of thte run-ID the
  // caller should eventually wait with the WaitForRun() API, to be sure that a
  // full function run happened after its Activate() call.
  size_t Activate();

  // Wait until a run-ID returned by the Activate() API completed. Returns the
  // value of the current run-ID. If such value or less or equal to run_id, the
  // wait did not complete successfully.
  size_t WaitForRun(size_t run_id);

 private:
  // Function implementing the main thread loop running the user function.
  void Runner();

  std::function<void()> function_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable run_cv_;
  size_t run_id_ = 0;
  size_t run_waiters_ = 0;
  size_t running_ = 0;
  bool activated_ = false;
  bool stopped_ = false;
  std::vector<std::unique_ptr<std::thread>> threads_;
};

}  // namespace util
}  // namespace xla

#endif  // X10_XLA_CLIENT_TRIGGERED_TASK_H_
