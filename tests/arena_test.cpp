#include "src/arena.h"

#include <folly/Expected.h>
#include <folly/FBVector.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <span>

#include "src/data/consts.h"
#include "src/data/structs.h"

class MemoryArenaTests : public ::testing::Test {};

using namespace recsys::knn_server;

TEST_F(MemoryArenaTests,
       MakeArenaFailsIfEmbeddingDimensionNotAMultipleOfSixteen) {
  size_t capacity = 10;
  size_t embedding_dim = 237;
  auto arena = MemoryArena::MakeArena(capacity, embedding_dim,
                                      EmbeddingDataType::Float32_t);
  ASSERT_FALSE(arena);
  EXPECT_EQ(arena.error(), StartupError::ArenaInvalidArgument);
}

TEST_F(MemoryArenaTests, MakeArenaReturnsErrorOnMMAPFailure) {
  // request 0 bytes from mmap
  size_t capacity = 0;
  size_t embedding_dim = 16;
  auto arena = MemoryArena::MakeArena(capacity, embedding_dim,
                                      EmbeddingDataType::Float32_t);
  ASSERT_FALSE(arena);
  EXPECT_EQ(arena.error(), StartupError::MMAPFailure);
}

TEST_F(MemoryArenaTests, MakeArenaSucceedsWithCorrectArguments) {
  size_t capacity = 10;       // positive number of embeddings
  size_t embedding_dim = 16;  // dimension is multiple of 16

  ASSERT_TRUE(MemoryArena::MakeArena(capacity, embedding_dim,
                                     EmbeddingDataType::Float32_t));
}

TEST_F(MemoryArenaTests, SetFloatEntryFailsWithInvalidEmbeddingDataType) {
  constexpr size_t capacity = 10;
  constexpr size_t embedding_dim = 16;
  EmbeddingDataType arena_type = EmbeddingDataType::Float64_t;

  folly::Expected<std::unique_ptr<MemoryArena>, StartupError> arena =
      MemoryArena::MakeArena(capacity, embedding_dim, arena_type);
  ASSERT_TRUE(arena);

  const float bad_query_vec[embedding_dim] = {};

  folly::Expected<folly::Unit, AllocError> set_entry_result;
  set_entry_result = (*arena)->SetEntry(0, bad_query_vec);

  ASSERT_FALSE(set_entry_result);
  EXPECT_EQ(set_entry_result.error(),
            AllocError::TypeDisagreesWithEmbeddingType);
}

TEST_F(MemoryArenaTests, SetDoubleEntryFailsWithInvalidEmbeddingDataType) {
  constexpr size_t capacity = 10;
  constexpr size_t embedding_dim = 16;
  EmbeddingDataType arena_type = EmbeddingDataType::Float32_t;

  folly::Expected<std::unique_ptr<MemoryArena>, StartupError> arena =
      MemoryArena::MakeArena(capacity, embedding_dim, arena_type);
  ASSERT_TRUE(arena);

  const double bad_query_vec[embedding_dim] = {};

  folly::Expected<folly::Unit, AllocError> set_entry_result;
  set_entry_result = (*arena)->SetEntry(0, bad_query_vec);

  ASSERT_FALSE(set_entry_result);
  EXPECT_EQ(set_entry_result.error(),
            AllocError::TypeDisagreesWithEmbeddingType);
}

template <typename T>
class MemoryArenaTypedTests : public ::testing::Test {};

using namespace recsys::knn_server;
using ::testing::FloatEq;
using ::testing::Pointwise;
using ImplementationType = ::testing::Types<float, double>;
TYPED_TEST_SUITE(MemoryArenaTypedTests, ImplementationType);

TYPED_TEST(MemoryArenaTypedTests, SetEntryFailsWithInvalidEmbeddingSize) {
  constexpr size_t capacity = 10;
  constexpr size_t embedding_dim = 16;
  constexpr size_t wrong_embedding_dim = 32;
  EmbeddingDataType arena_type;

  if constexpr (std::is_same_v<TypeParam, float>) {
    arena_type = EmbeddingDataType::Float32_t;
  } else {
    arena_type = EmbeddingDataType::Float64_t;
  }

  folly::Expected<std::unique_ptr<MemoryArena>, StartupError> arena =
      MemoryArena::MakeArena(capacity, embedding_dim, arena_type);
  ASSERT_TRUE(arena);

  const TypeParam bad_query_vec[wrong_embedding_dim] = {};

  folly::Expected<folly::Unit, AllocError> set_entry_result;
  set_entry_result = (*arena)->SetEntry(0, bad_query_vec);

  ASSERT_FALSE(set_entry_result);
  EXPECT_EQ(set_entry_result.error(), AllocError::IncorrectEmbeddingDim);
}

TYPED_TEST(MemoryArenaTypedTests,
           SetEntryFailsWhenRequestedIndexIsLargerThanCapacity) {
  constexpr size_t capacity = 1;
  constexpr size_t embedding_dim = 16;
  EmbeddingDataType arena_type;

  if constexpr (std::is_same_v<TypeParam, float>) {
    arena_type = EmbeddingDataType::Float32_t;
  } else {
    arena_type = EmbeddingDataType::Float64_t;
  }

  folly::Expected<std::unique_ptr<MemoryArena>, StartupError> arena =
      MemoryArena::MakeArena(capacity, embedding_dim, arena_type);
  ASSERT_TRUE(arena);

  const std::vector<TypeParam> vec0 = std::vector<TypeParam>(embedding_dim, 1);
  const std::vector<TypeParam> vec1 = std::vector<TypeParam>(embedding_dim, 2);
  ASSERT_TRUE((*arena)->SetEntry(0, vec0));

  folly::Expected<folly::Unit, AllocError> set_entry_result;
  set_entry_result = (*arena)->SetEntry(1, vec1);

  ASSERT_FALSE(set_entry_result);
  EXPECT_EQ(set_entry_result.error(), AllocError::RequestedIDAboveCapcity);
}

TYPED_TEST(MemoryArenaTypedTests, SetEntryCorrectlySetsEmbeddingEntry) {
  constexpr size_t capacity = 2;
  constexpr size_t embedding_dim = 16;
  EmbeddingDataType arena_type;

  if constexpr (std::is_same_v<TypeParam, float>) {
    arena_type = EmbeddingDataType::Float32_t;
  } else {
    arena_type = EmbeddingDataType::Float64_t;
  }

  folly::Expected<std::unique_ptr<MemoryArena>, StartupError> arena_status =
      MemoryArena::MakeArena(capacity, embedding_dim, arena_type);
  ASSERT_TRUE(arena_status);

  const std::unique_ptr<MemoryArena>& arena = arena_status.value();

  const std::vector<TypeParam> good_query_vec =
      std::vector<TypeParam>(embedding_dim, 1);

  ASSERT_TRUE(arena->SetEntry(0, good_query_vec));

  const folly::fbvector<unsigned long>& active_arena_ids =
      arena->GetActiveIds();
  ASSERT_EQ(active_arena_ids.size(), 1);
  EXPECT_EQ(active_arena_ids[0], 0);

  const folly::fbvector<bool>& id_status_map = arena->GetIdStatusMap();
  ASSERT_EQ(id_status_map.size(), capacity);
  EXPECT_TRUE(id_status_map[0]);
  EXPECT_FALSE(id_status_map[1]);

  const MemoryArena& arena_view = *arena;

  const TypeParam* arena_base_ptr = arena_view.GetArenaView<TypeParam>();
  const TypeParam* arena_embedding_entry =
      arena_base_ptr + active_arena_ids[0] * embedding_dim;

  EXPECT_THAT(std::vector<TypeParam>(arena_embedding_entry,
                                     arena_embedding_entry + embedding_dim),
              Pointwise(FloatEq(), good_query_vec));
}

TYPED_TEST(MemoryArenaTypedTests, SetEntryCorrectlySetsMultipleEmbeddingEntry) {
  constexpr size_t capacity = 3;
  constexpr size_t embedding_dim = 16;
  EmbeddingDataType arena_type;

  if constexpr (std::is_same_v<TypeParam, float>) {
    arena_type = EmbeddingDataType::Float32_t;
  } else {
    arena_type = EmbeddingDataType::Float64_t;
  }

  folly::Expected<std::unique_ptr<MemoryArena>, StartupError> arena_status =
      MemoryArena::MakeArena(capacity, embedding_dim, arena_type);
  ASSERT_TRUE(arena_status);

  const std::unique_ptr<MemoryArena>& arena = arena_status.value();

  const std::vector<TypeParam> good_query_vec_1 =
      std::vector<TypeParam>(embedding_dim, 1);

  std::vector<TypeParam> good_query_vec_2(embedding_dim);
  good_query_vec_2.at(0) = 1;

  ASSERT_TRUE(arena->SetEntry(0, good_query_vec_1));
  ASSERT_TRUE(arena->SetEntry(1, good_query_vec_2));

  const folly::fbvector<unsigned long>& active_arena_ids =
      arena->GetActiveIds();
  ASSERT_EQ(active_arena_ids.size(), 2);
  EXPECT_EQ(active_arena_ids[0], 0);
  EXPECT_EQ(active_arena_ids[1], 1);

  const folly::fbvector<bool>& id_status_map = arena->GetIdStatusMap();
  ASSERT_EQ(id_status_map.size(), capacity);

  EXPECT_TRUE(id_status_map[0]);
  EXPECT_TRUE(id_status_map[1]);
  EXPECT_FALSE(id_status_map[2]);

  const MemoryArena& arena_view = *arena;

  const TypeParam* arena_base_ptr = arena_view.GetArenaView<TypeParam>();
  const TypeParam* arena_embedding_entry_1 =
      arena_base_ptr + active_arena_ids[0] * embedding_dim;

  const TypeParam* arena_embedding_entry_2 =
      arena_base_ptr + active_arena_ids[1] * embedding_dim;

  EXPECT_THAT(std::vector<TypeParam>(arena_embedding_entry_1,
                                     arena_embedding_entry_1 + embedding_dim),
              Pointwise(FloatEq(), good_query_vec_1));

  EXPECT_THAT(std::vector<TypeParam>(arena_embedding_entry_2,
                                     arena_embedding_entry_2 + embedding_dim),
              Pointwise(FloatEq(), good_query_vec_2));
}
