#ifndef RECSYS_ENGINE_THREADS_THREADPOOL_H_
#define RECSYS_ENGINE_THREADS_THREADPOOL_H_

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>

namespace recsys {

class ThreadPool {
 public:
  ThreadPool(size_t num_threads = std::thread::hardware_concurrency());
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  auto Add(auto&& f, auto&&... args)
      -> std::future<std::invoke_result_t<decltype(f), decltype(args)...>> {
    using return_type = std::invoke_result_t<decltype(f), decltype(args)...>;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        [f = std::forward<decltype(f)>(f),
         ... args = std::forward<decltype(args)>(args)]() mutable {
          return f(args...);
        });

    std::future<return_type> res = task->get_future();

    {
      std::unique_lock<std::mutex> lock(mutex_lock_);
      task_queue_.emplace([task]() { (*task)(); });
      active_tasks_++;
    }
    cv_.notify_one();
    return res;
  }
  size_t GetPoolSize() {return pool_size_;}
  void WaitForAll();

 private:
  std::vector<std::thread> threads_;
  std::queue<std::function<void()>> task_queue_;
  std::condition_variable cv_;
  std::condition_variable main_thread_cv_;
  std::mutex mutex_lock_;
  size_t active_tasks_ = 0;
  size_t pool_size_ = 0;
  bool stop_ = false;
};

}  // namespace recsys

#endif  // RECSYS_ENGINE_THREADS_THREADPOOL_H_