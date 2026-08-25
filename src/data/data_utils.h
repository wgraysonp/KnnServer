#ifndef RECSYS_ENGINE_DATA_UTILS_H_
#define RECSYS_ENGINE_DATA_UTILS_H_

#include <cstddef>

#include "src/data/gen-cpp2/data_types.h"

namespace recsys::knn_server {

size_t GetTypeSize(EmbeddingDataType type);

}  // namespace recsys::knn_server
#endif  // RECSYS_ENGINE_DATA_UTILS_H_