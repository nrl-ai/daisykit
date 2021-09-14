#ifndef DAISYKIT_GRAPHS_CORE_CONCURRENT_QUEUE_H_
#define DAISYKIT_GRAPHS_CORE_CONCURRENT_QUEUE_H_

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

namespace daisykit {
namespace graphs {

template <typename T>
class ConcurrentQueue {
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