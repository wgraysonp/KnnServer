#include "src/arena.h"

#include <gtest/gtest.h>

class MemoryArenaTests : public ::testing::Test {};

using namespace recsys;

TEST_F(MemoryArenaTests,
       MakeArenaFailsIfEmbeddingDimensionNotAMultipleOfSixteen) {
  size_t capacity = 10;
  size_t embedding_dim = 237;
  auto arena =
      MemoryArena::MakeArena(10, embedding_dim, EmbeddingDataType::Float32_t);
  ASSERT_FALSE(arena);
  EXPECT_EQ(arena.error(), StartupError::ArenaInvalidArgument);
}