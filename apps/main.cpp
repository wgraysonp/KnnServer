#include <folly/Expected.h>
#include <folly/FBVector.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>

#include "src/arena.h"
#include "src/data/types.h"
#include "src/macros.h"
#include "src/search.h"
#include "src/service/pipelines.h"

using namespace recsys;

int main() {
  std::cout << "Initializing memory arena via Cmake configuration..."
            << std::endl;

  size_t item_count = 1000000;

  std::unique_ptr<MemoryArena> arena;

  CHECK_OK_AND_ASSIGN(arena,
                      StartServiceAndCreateArena(item_count, EMBEDDING_DIM,
                                                 EmbeddingDataType::Float32_t));

  std::cout << "Initialization complete. Starting search" << std::endl;

  folly::fbvector<float> query_vector(EMBEDDING_DIM, 0.5f);

  auto start = std::chrono::high_resolution_clock::now();

  folly::fbvector<EmbeddingSearchResult> result =
      FindNClosest<float>(*arena, query_vector, 10);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;

  std::cout << "KNN Search took: " << duration.count() << " ms\n";

  printf("Number of Items: %lu\n", result.size());
  for (const auto& item : result) {
    printf("Vector id: %lu\n", item.id);
  }

  return 0;
}