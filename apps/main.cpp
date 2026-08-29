#include <folly/Expected.h>
#include <folly/FBVector.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>

#include "src/arena.h"
#include "src/data/consts.h"
#include "src/data/gen-cpp2/data_types.h"
#include "src/macros.h"
#include "src/search.h"
#include "src/service/gen-cpp2/service_types.h"
#include "src/service/pipelines.h"

using namespace recsys::knn_server;

int main() {
  std::cout << "Initializing memory arena..." << std::endl;

  size_t item_count = 1000000;

  std::unique_ptr<MemoryArena> arena;

  CHECK_OK_AND_ASSIGN(arena,
                      StartServiceAndCreateArena(item_count, EMBEDDING_DIM,
                                                 EmbeddingDataType::Float32_t));

  std::cout << "Initialization complete. Starting search" << std::endl;

  std::vector<float> query_vector(EMBEDDING_DIM, 0.5f);

  QueryRequest request;
  request.raw_query_vector_ref() =
      std::string(reinterpret_cast<const char*>(query_vector.data()),
                  query_vector.size() * sizeof(float));
  request.type_ref() = EmbeddingDataType::Float32_t;
  request.library_id_ref() = 1;
  request.embedding_dim_ref() = 128;
  request.n_closest_ref() = 10;

  QueryResponse response;

  CHECK_OK(ProcessRequestAndPopulateResponse(*arena, request, response));

  std::cout << "KNN Search took: "
            << response.milli_seconds_passed_ref().value() << " ms\n";

  printf("Number of Items: %lu\n", response.results_ref()->size());
  for (const auto& item : *response.results_ref()) {
    printf("Vector id: %lu\n", item.id_ref().value());
  }

  return 0;
}