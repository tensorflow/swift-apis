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

#ifndef X10_XLA_CLIENT_THREAD_POOL_H_
#define X10_XLA_CLIENT_THREAD_POOL_H_

#include <functional>
#include <memory>
#include <thread>

namespace xla {
namespace env {

class Completion {
 public:
  class Data;

  explicit Completion(std::shared_ptr<Data> data);

  ~Completion();

  void Wait();

 private:
  std::shared_ptr<Data> data_;
};

// Schedules a closure to be run. The closure should not block waiting for other
// events.
void ScheduleClosure(std::function<void()> closure);
Completion ScheduleClosureWithCompletion(std::function<void()> closure);

// Schedules a closure which might wait for IO or other events/conditions.
void ScheduleIoClosure(std::function<void()> closure);
Completion ScheduleIoClosureWithCompletion(std::function<void()> closure);

}  // namespace env
}  // namespace xla

#endif  // X10_XLA_CLIENT_THREAD_POOL_H_
