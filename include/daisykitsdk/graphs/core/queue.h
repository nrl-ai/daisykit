// Copyright 2021 The DaisyKit Authors.
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

#ifndef DAISYKIT_GRAPHS_CORE_QUEUE_H_
#define DAISYKIT_GRAPHS_CORE_QUEUE_H_

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

namespace daisykit {
namespace graphs {

/// Thread-safe queue class for data.
template <typename T>
class Queue {
 public:
  T WaitPop() {
    std::unique_lock<std::mutex> mlock(mutex_);
    while (queue_.empty()) {
      cond_.wait(mlock);
    }
    auto item = queue_.front();
    queue_.pop();
    return item;
  }

  void WaitPop(T& item) {
    std::unique_lock<std::mutex> mlock(mutex_);
    while (queue_.empty()) {
      cond_.wait(mlock);
    }
    item = queue_.front();
    queue_.pop();
  }

  int Pop(T& item) {
    std::unique_lock<std::mutex> mlock(mutex_);
    if (queue_.empty()) {
      return -1;
    }
    item = queue_.front();
    queue_.pop();
    return 0;
  }

  void Push(const T& item) {
    std::unique_lock<std::mutex> mlock(mutex_);
    queue_.push(item);
    mlock.unlock();
    cond_.notify_one();
  }

  void Push(T&& item) {
    std::unique_lock<std::mutex> mlock(mutex_);
    queue_.push(std::move(item));
    mlock.unlock();
    cond_.notify_one();
  }

  size_t Size() {
    std::unique_lock<std::mutex> mlock(mutex_);
    return queue_.size();
  }

  void WaitForData() {
    std::unique_lock<std::mutex> mlock(mutex_);
    cond_.wait(mlock, [this] { return !queue_.empty(); });
  }

 private:
  std::queue<T> queue_;
  std::mutex mutex_;
  std::condition_variable cond_;
};

}  // namespace graphs
}  // namespace daisykit

#endif