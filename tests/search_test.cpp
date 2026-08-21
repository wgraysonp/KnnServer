#include "src/search.h"

#include <folly/Expected.h>
#include <folly/FBVector.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <memory>

#include "src/arena.h"
#include "src/data/structs.h"
#include "src/data/types.h"

using namespace recsys;

class SearchTests : public ::testing::Test {
 protected:
  std::unique_ptr<MemoryArena> arena;
  EmbeddingDataType type = EmbeddingDataType::Float32_t;
  size_t embedding_dim = 16;
  size_t capacity = 100;

  void SetUp() override {
    auto arena_result = MemoryArena::MakeArena(capacity, embedding_dim, type);
    if (!arena_result) {
      FAIL();
    }
    arena = std::move(*arena_result);
  }
};

TEST_F(SearchTests, FindNClosestFindsTheCloserOfTwoEmbeddings) {
  arena->SetEntry(0, std::vector<float>(embedding_dim, 0.5f));
  arena->SetEntry(1, std::vector<float>(embedding_dim, 0.6f));

  folly::fbvector<float> query_vec =
      folly::fbvector<float>(embedding_dim, 0.6f);

  folly::fbvector<EmbeddingSearchResult> res =
      FindNClosest<float>(*arena, query_vec, 1);

  EmbeddingSearchResult expected_closest =
      EmbeddingSearchResult{.id = 1, .dist = 0};

  ASSERT_EQ(res.size(), 1);
  ASSERT_EQ(res[0].id, expected_closest.id);
  ASSERT_EQ(res[0].dist, expected_closest.dist);
}