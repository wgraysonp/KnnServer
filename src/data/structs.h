#ifndef RECSYS_ENGINE_STRUCTS_H_
#define RECSYS_ENGINE_STRUCTS_H_

#include <folly/FBVector.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/data/gen-cpp2/data_types.h"

namespace recsys::knn_server {

struct EmbeddingLibrary {
  EmbeddingDataType type;
  size_t library_id;
  size_t embedding_dim;
};

}  // namespace recsys::knn_server

#endif  // RECSYS_ENGINE_STRUCTS_H_