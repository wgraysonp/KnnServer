#include "threadpool.h"

#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

namespace recsys::knn_server {
ThreadPool::ThreadPool(size_t num_threads) : pool_size_(num_threads) {
  for (size_t i = 0; i < num_threads; ++i) {
    threads_.emplace_back([this]() {
      while (true) {
        std::function<void()> task;
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
          if (active_tasks_ == 0) {
            main_thread_cv_.notify_all();
          }
        }
      }
    });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(mutex_lock_);
    stop_ = true;
    cv_.notify_all();
  }
  for (auto& thread : threads_) {
    thread.join();
  }
}

void ThreadPool::WaitForAll() {
  {
    std::unique_lock<std::mutex> lock(mutex_lock_);
    main_thread_cv_.wait(lock, [this]() {
      return (active_tasks_ == 0) && (task_queue_.empty());
    });
  }
}
}  // namespace recsys::knn_server