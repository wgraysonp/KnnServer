#include <folly/futures/Future.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <exception>

#include "src/threads/custom_folly_pool.h"

using namespace recsys::knn_server;

class FollyPoolTests : public ::testing::Test {
 protected:
  FollyPool bundle;

  FollyPoolTests() : bundle() {};
};

TEST_F(FollyPoolTests, ThreadsUpdateSharedVariableCorrectly) {
  std::atomic<int> shared_counter(0);
  const int num_tasks = 100000;

  std::vector<folly::Future<folly::Unit>> tasks;

  for (int i = 0; i < num_tasks; ++i) {
    tasks.emplace_back(
        folly::via(&bundle, [&shared_counter]() { shared_counter++; }));
  }

  folly::collect(tasks).get();

  EXPECT_EQ(shared_counter, 100000);
}

TEST_F(FollyPoolTests, FollyExceptionsAreIsolated) {
  std::atomic<int> standard_counter(0);

  folly::Future<folly::Unit> bad_future = folly::via(
      &bundle, []() { throw std::runtime_error("Intentional Exception"); });

  folly::Future<folly::Unit> good_future =
      folly::via(&bundle, [&standard_counter]() { standard_counter++; });

  bundle.WaitForAll();
  EXPECT_EQ(standard_counter, 1);
  EXPECT_THROW(std::move(bad_future).get(), std::runtime_error);
}

TEST_F(FollyPoolTests, FollysDoBasicTaskAndReturnValueCorrectly) {
  std::vector<int> vec1 = {2, 3, 1};
  std::vector<int> vec2 = {1, 3, 2};
  std::vector<int> vec3 = {3, 2, 1};

  std::vector<folly::Future<std::vector<int>>> results_futures;

  results_futures.emplace_back(folly::via(&bundle, [&vec1]() {
    std::sort(vec1.begin(), vec1.end());
    std::vector<int> res;
    for (auto x : vec1) {
      res.push_back(x * 2);
    }
    return res;
  }));

  results_futures.emplace_back(folly::via(&bundle, [&vec2]() {
    std::sort(vec2.begin(), vec2.end());
    std::vector<int> res;
    for (auto x : vec2) {
      res.push_back(x * 2);
    }
    return res;
  }));

  results_futures.emplace_back(folly::via(&bundle, [&vec3]() {
    std::sort(vec3.begin(), vec3.end());
    std::vector<int> res;
    for (auto x : vec3) {
      res.push_back(x * 2);
    }
    return res;
  }));

  std::vector<int> sorted_vec = {2, 4, 6};

  std::vector<std::vector<int>> results = folly::collect(results_futures).get();

  ASSERT_EQ(results.size(), 3);

  EXPECT_EQ(results.at(0), sorted_vec);
  EXPECT_EQ(results.at(1), sorted_vec);
  EXPECT_EQ(results.at(2), sorted_vec);
}