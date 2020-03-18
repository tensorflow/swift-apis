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

#ifndef X10_XLA_CLIENT_ASYNC_TASK_H_
#define X10_XLA_CLIENT_ASYNC_TASK_H_

#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/thread_pool.h"

namespace xla {
namespace util {

template <typename T>
class AsyncTask {
  struct Data {
    Data(std::function<T()> taskfn) : taskfn(std::move(taskfn)) {}

    std::function<T()> taskfn;
    std::mutex mutex;
    std::condition_variable cv;
    bool scheduled = false;
    bool completed = false;
    absl::optional<T> result;
    std::exception_ptr exptr;
  };

 public:
  explicit AsyncTask(std::function<T()> taskfn)
      : data_(std::make_shared<Data>(std::move(taskfn))) {}

  AsyncTask& Wait() {
    std::unique_lock<std::mutex> lock(data_->mutex);
    XLA_CHECK(data_->scheduled);
    data_->cv.wait(lock, [this] { return data_->completed; });
    if (data_->exptr != nullptr) {
      std::rethrow_exception(data_->exptr);
    }
    return *this;
  }

  AsyncTask& Schedule() {
    auto completer = [data = data_]() {
      absl::optional<T> result;
      std::exception_ptr exptr;
      try {
        result = data->taskfn();
      } catch (...) {
        exptr = std::current_exception();
      }

      std::lock_guard<std::mutex> lock(data->mutex);
      if (result) {
        data->result = std::move(*result);
      } else {
        data->exptr = std::move(exptr);
      }
      data->completed = true;
      data->cv.notify_all();
    };

    {
      std::lock_guard<std::mutex> lock(data_->mutex);
      XLA_CHECK(!data_->scheduled);
      data_->scheduled = true;
    }
    xla::env::ScheduleIoClosure(std::move(completer));
    return *this;
  }

  const T& GetValue() const {
    std::lock_guard<std::mutex> lock(data_->mutex);
    return *data_->result;
  }

  T ConsumeValue() {
    std::lock_guard<std::mutex> lock(data_->mutex);
    return std::move(*data_->result);
  }

 private:
  std::shared_ptr<Data> data_;
};

}  // namespace util
}  // namespace xla

#endif  // X10_XLA_CLIENT_ASYNC_TASK_H_
