#include "src/threads/custom_folly_pool.h"

#include <folly/Executor.h>
#include <folly/Function.h>

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <vector>

namespace recsys {

FollyPool::FollyPool(size_t num_threads) : pool_size_(num_threads) {
  for (size_t i = 0; i < num_threads; ++i) {
    threads_.emplace_back([this]() { WorkerLoop(); });
  }
}

FollyPool::~FollyPool() {
  {
    std::unique_lock<std::mutex> lock(mutex_lock_);
    stop_ = true;
    cv_.notify_all();
  }
  for (auto& thread : threads_) {
    thread.join();
  }
}

void FollyPool::add(folly::Func task) {
  {
    std::unique_lock<std::mutex> lock(mutex_lock_);
    task_queue_.push(std::move(task));
    active_tasks_++;
  }
  cv_.notify_one();
}

void FollyPool::WorkerLoop() {
  while (true) {
    folly::Func task;
    {
      std::unique_lock<std::mutex> lock(mutex_lock_);
      cv_.wait(lock, [this]() { return !task_queue_.empty() || stop_; });
      if (stop_ && task_queue_.empty()) return;

      task = std::move(task_queue_.front());
      task_queue_.pop();
    }
    task();
    {
      std::unique_lock<std::mutex> lock(mutex_lock_);
      active_tasks_--;
      if (active_tasks_ == 0 && task_queue_.empty()) {
        main_thread_cv_.notify_all();
      }
    }
  }
}

void FollyPool::WaitForAll() {
  {
    std::unique_lock<std::mutex> lock(mutex_lock_);
    main_thread_cv_.wait(lock, [this]() {
      return (active_tasks_ == 0) && (task_queue_.empty());
    });
  }
}

}  // namespace recsys