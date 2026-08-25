#include "src/threads/threadpool.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <exception>

using namespace recsys::knn_server;

class ThreadPoolTests : public ::testing::Test {};

TEST_F(ThreadPoolTests, ThreadsUpdateSharedVariableCorrectly) {
  auto bundle = ThreadPool();
  std::atomic<int> shared_counter(0);
  const int num_tasks = 100000;

  for (int i = 0; i < num_tasks; ++i) {
    bundle.Add([&]() -> void { shared_counter++; });
  }

  bundle.WaitForAll();
  EXPECT_EQ(shared_counter, 100000);
}

TEST_F(ThreadPoolTests, ThreadExceptionsAreIsolated) {
  auto bundle = ThreadPool(2);
  std::atomic<int> standard_counter(0);

  auto bad_future =
      bundle.Add([]() { throw std::runtime_error("Intentional Exception"); });

  auto good_future = bundle.Add([&standard_counter]() { standard_counter++; });

  bundle.WaitForAll();
  EXPECT_EQ(standard_counter, 1);
  EXPECT_THROW(bad_future.get(), std::runtime_error);
}

TEST_F(ThreadPoolTests, ThreadsDoBasicTaskAndReturnValueCorrectly) {
  auto bundle = ThreadPool(3);

  std::vector<int> vec1 = {2, 3, 1};
  std::vector<int> vec2 = {1, 3, 2};
  std::vector<int> vec3 = {3, 2, 1};

  auto fut_1 = bundle.Add([&vec1]() {
    std::sort(vec1.begin(), vec1.end());
    std::vector<int> res;
    for (auto x : vec1) {
      res.push_back(x * 2);
    }
    return res;
  });

  auto fut_2 = bundle.Add([&vec2]() {
    std::sort(vec2.begin(), vec2.end());
    std::vector<int> res;
    for (auto x : vec2) {
      res.push_back(x * 2);
    }
    return res;
  });

  auto fut_3 = bundle.Add([&vec3]() {
    std::sort(vec3.begin(), vec3.end());
    std::vector<int> res;
    for (auto x : vec3) {
      res.push_back(x * 2);
    }
    return res;
  });

  auto res1 = fut_1.get();
  auto res2 = fut_2.get();
  auto res3 = fut_3.get();

  std::vector<int> sorted_vec = {2, 4, 6};

  EXPECT_EQ(res1, sorted_vec);
  EXPECT_EQ(res2, sorted_vec);
  EXPECT_EQ(res3, sorted_vec);
}