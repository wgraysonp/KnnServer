#ifndef RECSYS_ENGINE_EMBEDDING_LIBRARY_H_
#define RECSYS_ENGINE_EMBEDDING_LIBRARY_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/consts.h"

namespace recsys {

struct QueryRequest {
  size_t library_id;
  size_t embedding_dim;
  EmbeddingDataType type;
  std::vector<uint8_t> query_vector;
};

struct QueryResponse {
  bool success;
};

struct EmbeddingLibrary {
  size_t library_id;
  size_t embedding_dim;
  EmbeddingDataType type;
};

} // namespace recsys

#endif // RECSYS_ENGINE_EMBEDDING_LIBRARY_H_