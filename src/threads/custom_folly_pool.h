#ifndef RECSYS_ENGINE_THREADS_CUSTOM_FOLLY_POOL_H_
#define RECSYS_ENGINE_THREADS_CUSTOM_FOLLY_POOL_H_

#include <folly/Executor.h>
#include <folly/Function.h>

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace recsys {
class FollyPool : public folly::Executor {
 public:
  FollyPool(size_t num_threads = std::thread::hardware_concurrency());
  ~FollyPool() override;
  FollyPool(const FollyPool&) = delete;
  FollyPool& operator=(const FollyPool&) = delete;

  void add(folly::Func task) override;

  void WaitForAll();

 private:
  void WorkerLoop();
  std::vector<std::thread> threads_;
  std::queue<folly::Func> task_queue_;
  std::condition_variable cv_;
  std::condition_variable main_thread_cv_;
  std::mutex mutex_lock_;
  size_t active_tasks_ = 0;
  size_t pool_size_ = 0;
  bool stop_ = false;
};

}  // namespace recsys

#endif  // RECSYS_ENGINE_THREADS_CUSTOM_FOLLY_POOL_H_