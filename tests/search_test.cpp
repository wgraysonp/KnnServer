#include "src/search.h"

#include <folly/Expected.h>
#include <folly/FBVector.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <type_traits>

#include "src/arena.h"
#include "src/data/structs.h"
#include "src/data/types.h"

using namespace recsys;
using ::testing::FloatEq;
using ::testing::Pointwise;
using ImplementationType = ::testing::Types<float, double>;

// 1. Map your acceptable epsilon thresholds for each datatype
template <typename T>
struct TestTolerance;

template <>
struct TestTolerance<double> {
  static constexpr double value = 1e-6;
};
template <>
struct TestTolerance<float> {
  static constexpr float value = 1e-4f;
};

MATCHER_P(IsNearlyEqual, expected, "") {
  using T = std::decay_t<decltype(expected)>;

  double diff =
      std::abs(static_cast<double>(arg) - static_cast<double>(expected));
  double allowed_epsilon = static_cast<double>(TestTolerance<T>::value);

  return diff <= allowed_epsilon;
}

template <typename T>
struct EmbeddingTypeMap;

template <>
struct EmbeddingTypeMap<float> {
  static constexpr EmbeddingDataType value = EmbeddingDataType::Float32_t;
};
template <>
struct EmbeddingTypeMap<double> {
  static constexpr EmbeddingDataType value = EmbeddingDataType::Float64_t;
};

template <typename T>
class SearchTests : public ::testing::Test {
 protected:
  std::unique_ptr<MemoryArena> arena;
  EmbeddingDataType type;
  size_t embedding_dim = 16;
  size_t capacity = 100;

  void SetUp() override {
    type = EmbeddingTypeMap<T>::value;

    auto arena_result = MemoryArena::MakeArena(capacity, embedding_dim, type);
    if (!arena_result) {
      FAIL();
    }
    arena = std::move(*arena_result);
  }
};

TYPED_TEST_SUITE(SearchTests, ImplementationType);

TYPED_TEST(SearchTests, FindNClosestFindsTheCloserOfFourEmbeddings) {
  using Base = SearchTests<TypeParam>;
  ASSERT_TRUE(Base::arena->SetEntry(
      0, std::vector<TypeParam>(Base::embedding_dim,
                                static_cast<TypeParam>(0.5))));
  ASSERT_TRUE(Base::arena->SetEntry(
      1, std::vector<TypeParam>(Base::embedding_dim,
                                static_cast<TypeParam>(0.6))));
  ASSERT_TRUE(Base::arena->SetEntry(
      2, std::vector<TypeParam>(Base::embedding_dim,
                                static_cast<TypeParam>(0.8))));
  ASSERT_TRUE(Base::arena->SetEntry(
      3, std::vector<TypeParam>(Base::embedding_dim,
                                static_cast<TypeParam>(0.7))));

  size_t n_workers = 1;
  size_t n_closest = 1;

  folly::fbvector<TypeParam> query_vec = folly::fbvector<TypeParam>(
      Base::embedding_dim, static_cast<TypeParam>(0.6));

  folly::fbvector<EmbeddingSearchResult> res =
      FindNClosest<TypeParam>(*Base::arena, query_vec, n_closest, n_workers);

  EmbeddingSearchResult expected_closest =
      EmbeddingSearchResult{.id = 1, .dist = 0};

  ASSERT_EQ(res.size(), 1);
  ASSERT_EQ(res[0].id, expected_closest.id);
  ASSERT_THAT(res[0].dist, IsNearlyEqual(expected_closest.dist));
}

TYPED_TEST(SearchTests,
           FindNClosestSuceedsWhenWorkerChunkSizeIsNotMultipleOfFour) {
  using Base = SearchTests<TypeParam>;

  ASSERT_TRUE(Base::arena->SetEntry(
      0, std::vector<TypeParam>(Base::embedding_dim,
                                static_cast<TypeParam>(0.5f))));
  ASSERT_TRUE(Base::arena->SetEntry(
      1, std::vector<TypeParam>(Base::embedding_dim,
                                static_cast<TypeParam>(0.6f))));
  ASSERT_TRUE(Base::arena->SetEntry(
      3, std::vector<TypeParam>(Base::embedding_dim,
                                static_cast<TypeParam>(0.8f))));

  size_t n_workers = 1;
  size_t n_closest = 1;

  folly::fbvector<TypeParam> query_vec =
      folly::fbvector<TypeParam>(Base::embedding_dim, 0.6f);

  folly::fbvector<EmbeddingSearchResult> res =
      FindNClosest<TypeParam>(*Base::arena, query_vec, n_closest, n_workers);

  EmbeddingSearchResult expected_closest =
      EmbeddingSearchResult{.id = 1, .dist = 0};

  ASSERT_EQ(res.size(), 1);
  EXPECT_EQ(res[0].id, expected_closest.id);
  EXPECT_THAT(res[0].dist, IsNearlyEqual(expected_closest.dist));
}

TYPED_TEST(
    SearchTests,
    FindNClosestSuceedsWhenNumberOfWorkersIsLargerThanNumberOfEmbeddings) {
  using Base = SearchTests<TypeParam>;

  ASSERT_TRUE(Base::arena->SetEntry(
      0, std::vector<TypeParam>(Base::embedding_dim,
                                static_cast<TypeParam>(0.5f))));
  ASSERT_TRUE(Base::arena->SetEntry(
      1, std::vector<TypeParam>(Base::embedding_dim,
                                static_cast<TypeParam>(0.6f))));
  ASSERT_TRUE(Base::arena->SetEntry(
      2, std::vector<TypeParam>(Base::embedding_dim,
                                static_cast<TypeParam>(0.8f))));
  ASSERT_TRUE(Base::arena->SetEntry(
      3, std::vector<TypeParam>(Base::embedding_dim,
                                static_cast<TypeParam>(0.9f))));

  size_t n_workers = 5;
  size_t n_closest = 1;

  folly::fbvector<TypeParam> query_vec =
      folly::fbvector<TypeParam>(Base::embedding_dim, 0.6f);

  folly::fbvector<EmbeddingSearchResult> res =
      FindNClosest<TypeParam>(*Base::arena, query_vec, n_closest, n_workers);

  EmbeddingSearchResult expected_closest =
      EmbeddingSearchResult{.id = 1, .dist = 0};

  ASSERT_EQ(res.size(), 1);
  EXPECT_EQ(res[0].id, expected_closest.id);
  EXPECT_THAT(res[0].dist, IsNearlyEqual(expected_closest.dist));
}

TYPED_TEST(
    SearchTests,
    FindNClosestSuceedsWhenNumberOfNeighborsIsLargerThanNumberOfEmbeddings) {
  using Base = SearchTests<TypeParam>;

  std::vector<TypeParam> embedding_0 =
      std::vector<TypeParam>(Base::embedding_dim, static_cast<TypeParam>(0.5f));
  std::vector<TypeParam> embedding_1 =
      std::vector<TypeParam>(Base::embedding_dim, static_cast<TypeParam>(0.6f));
  std::vector<TypeParam> embedding_2 =
      std::vector<TypeParam>(Base::embedding_dim, static_cast<TypeParam>(0.8f));
  std::vector<TypeParam> embedding_3 =
      std::vector<TypeParam>(Base::embedding_dim, static_cast<TypeParam>(0.9f));

  ASSERT_TRUE(Base::arena->SetEntry(0, embedding_0));
  ASSERT_TRUE(Base::arena->SetEntry(1, embedding_1));
  ASSERT_TRUE(Base::arena->SetEntry(2, embedding_2));
  ASSERT_TRUE(Base::arena->SetEntry(3, embedding_3));

  size_t n_workers = 1;
  size_t n_closest = 5;

  folly::fbvector<TypeParam> query_vec =
      folly::fbvector<TypeParam>(Base::embedding_dim, 0.6f);

  folly::fbvector<EmbeddingSearchResult> res =
      FindNClosest<TypeParam>(*Base::arena, query_vec, n_closest, n_workers);

  // expected order
  // 1. embedding_1 - distance 0
  // 2. embedding_0 - distance 16*0.1^2
  // 3. embedding_2 - distance 16*0.2^2
  // 4. embedding_3 - distance 16*0.3^2

  EmbeddingSearchResult expected_1 = EmbeddingSearchResult{.id = 1, .dist = 0};
  EmbeddingSearchResult expected_2 =
      EmbeddingSearchResult{.id = 0, .dist = 16 * 0.1 * 0.1};
  EmbeddingSearchResult expected_3 =
      EmbeddingSearchResult{.id = 2, .dist = 16 * 0.2 * 0.2};
  EmbeddingSearchResult expected_4 =
      EmbeddingSearchResult{.id = 3, .dist = 16 * 0.3 * 0.3};

  ASSERT_EQ(res.size(), 4);

  EXPECT_EQ(res[0].id, expected_1.id);
  EXPECT_THAT(res[0].dist, IsNearlyEqual(expected_1.dist));

  EXPECT_EQ(res[1].id, expected_2.id);
  EXPECT_THAT(res[1].dist, IsNearlyEqual(expected_2.dist));

  EXPECT_EQ(res[2].id, expected_3.id);
  EXPECT_THAT(res[2].dist, IsNearlyEqual(expected_3.dist));

  EXPECT_EQ(res[3].id, expected_4.id);
  EXPECT_THAT(res[3].dist, IsNearlyEqual(expected_4.dist));
}

TYPED_TEST(
    SearchTests,
    FindNClosestSuceedsWhenNumberOfNeighborsIsSmallerThanNumberOfEmbeddings) {
  using Base = SearchTests<TypeParam>;

  std::vector<TypeParam> embedding_0 =
      std::vector<TypeParam>(Base::embedding_dim, static_cast<TypeParam>(0.5f));
  std::vector<TypeParam> embedding_1 =
      std::vector<TypeParam>(Base::embedding_dim, static_cast<TypeParam>(0.6f));
  std::vector<TypeParam> embedding_2 =
      std::vector<TypeParam>(Base::embedding_dim, static_cast<TypeParam>(0.8f));
  std::vector<TypeParam> embedding_3 =
      std::vector<TypeParam>(Base::embedding_dim, static_cast<TypeParam>(0.9f));

  std::vector<TypeParam> too_far_embedding =
      std::vector<TypeParam>(Base::embedding_dim, static_cast<TypeParam>(1));

  ASSERT_TRUE(Base::arena->SetEntry(0, embedding_0));
  ASSERT_TRUE(Base::arena->SetEntry(1, embedding_1));
  ASSERT_TRUE(Base::arena->SetEntry(2, embedding_2));
  ASSERT_TRUE(Base::arena->SetEntry(3, embedding_3));

  ASSERT_TRUE(Base::arena->SetEntry(4, too_far_embedding));
  ASSERT_TRUE(Base::arena->SetEntry(5, too_far_embedding));
  ASSERT_TRUE(Base::arena->SetEntry(6, too_far_embedding));
  ASSERT_TRUE(Base::arena->SetEntry(7, too_far_embedding));
  ASSERT_TRUE(Base::arena->SetEntry(8, too_far_embedding));
  ASSERT_TRUE(Base::arena->SetEntry(9, too_far_embedding));

  size_t n_workers = 1;
  size_t n_closest = 4;  // Should capture embeddings 0-3

  folly::fbvector<TypeParam> query_vec =
      folly::fbvector<TypeParam>(Base::embedding_dim, 0.6f);

  folly::fbvector<EmbeddingSearchResult> res =
      FindNClosest<TypeParam>(*Base::arena, query_vec, n_closest, n_workers);

  // expected order
  // 1. embedding_1 - distance 0
  // 2. embedding_0 - distance 16*0.1^2
  // 3. embedding_2 - distance 16*0.2^2
  // 4. embedding_3 - distance 16*0.3^2

  EmbeddingSearchResult expected_1 = EmbeddingSearchResult{.id = 1, .dist = 0};
  EmbeddingSearchResult expected_2 =
      EmbeddingSearchResult{.id = 0, .dist = 16 * 0.1 * 0.1};
  EmbeddingSearchResult expected_3 =
      EmbeddingSearchResult{.id = 2, .dist = 16 * 0.2 * 0.2};
  EmbeddingSearchResult expected_4 =
      EmbeddingSearchResult{.id = 3, .dist = 16 * 0.3 * 0.3};

  ASSERT_EQ(res.size(), 4);

  EXPECT_EQ(res[0].id, expected_1.id);
  EXPECT_THAT(res[0].dist, IsNearlyEqual(expected_1.dist));

  EXPECT_EQ(res[1].id, expected_2.id);
  EXPECT_THAT(res[1].dist, IsNearlyEqual(expected_2.dist));

  EXPECT_EQ(res[2].id, expected_3.id);
  EXPECT_THAT(res[2].dist, IsNearlyEqual(expected_3.dist));

  EXPECT_EQ(res[3].id, expected_4.id);
  EXPECT_THAT(res[3].dist, IsNearlyEqual(expected_4.dist));
}