#ifndef RECSYS_ENGINE_DATA_CONSTS_H_
#define RECSYS_ENGINE_DATA_CONSTS_H_

#include <cstddef>

namespace recsys::knn_server {
constexpr size_t EMBEDDING_DIM = 128;
constexpr size_t BATCH_SIZE = 256;
constexpr size_t NUM_NEON_PIPELINES = 4;

enum class InvalidArgmentError {
  EnumTypeUndefined,
  TypeDisagreesWithArenaType,
};

enum class SearchRequestError {
  InvalidRawDataSizeError,
  InvalidSearchLibraryError,
};

enum class AllocError {
  BadAlloc,
  RequestedIDAboveCapcity,
  TypeDisagreesWithEmbeddingType,
  IncorrectEmbeddingDim
};

enum class StartupError {
  ArenaInvalidArgument,
  MMAPFailure,
  DataLoadError,
  Unknown
};

}  // namespace recsys::knn_server
#endif  // RECSYS_ENGINE_DATA_CONSTS_H_