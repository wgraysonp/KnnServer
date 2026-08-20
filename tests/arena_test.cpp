#include "src/arena.h"

#include <folly/Expected.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <span>

#include "src/data/structs.h"
#include "src/data/types.h"

class MemoryArenaTests : public ::testing::Test {};

using namespace recsys;

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

template<typename T>
class MemoryArenaTypedTests : public ::testing::Test {};

using namespace recsys;
using ImplementationType = ::testing::Types<float, double>;
TYPED_TEST_SUITE(MemoryArenaTypedTests, ImplementationType);

TYPED_TEST(MemoryArenaTypedTests, SetEntryFailsWithInvalidEmbeddingSize) {
  constexpr size_t capacity = 10;
  constexpr size_t embedding_dim = 16;
  constexpr size_t wrong_embedding_dim = 32;
  EmbeddingDataType arena_type;

  if constexpr (std::is_same_v<TypeParam, float>) {
    arena_type = EmbeddingDataType::Float32_t;
  }
  else {
    arena_type = EmbeddingDataType::Float64_t;
  }

  folly::Expected<std::unique_ptr<MemoryArena>, StartupError> arena =
      MemoryArena::MakeArena(capacity, embedding_dim, arena_type);
  ASSERT_TRUE(arena);

  const TypeParam bad_query_vec[wrong_embedding_dim] = {};

  folly::Expected<folly::Unit, AllocError> set_entry_result;
  set_entry_result = (*arena)->SetEntry(0, bad_query_vec);

  ASSERT_FALSE(set_entry_result);
  EXPECT_EQ(set_entry_result.error(),
            AllocError::IncorrectEmbeddingDim);
}
