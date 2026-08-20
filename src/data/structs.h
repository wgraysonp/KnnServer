#ifndef RECSYS_ENGINE_STRUCTS_H_
#define RECSYS_ENGINE_STRUCTS_H_

#include <folly/FBVector.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/data/types.h"

namespace recsys {

struct EmbeddingSearchResult {
  unsigned long id;
  double dist;
};

struct QueryRequest {
  folly::fbvector<uint8_t> query_vector;
  EmbeddingDataType type;
  size_t library_id;
  size_t embedding_dim;
  size_t n_closest;
  bool include_vectors = false;
};

struct QueryResponse {
  ResponseStatus status;
  std::optional<uint64_t> micros_passed = std::nullopt;
  std::optional<folly::fbvector<EmbeddingSearchResult>> nearest_neighbors =
      std::nullopt;
};

struct EmbeddingLibrary {
  EmbeddingDataType type;
  size_t library_id;
  size_t embedding_dim;
};

}  // namespace recsys

#endif  // RECSYS_ENGINE_STRUCTS_H_