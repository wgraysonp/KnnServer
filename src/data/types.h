#ifndef RECSYS_ENGINE_DATA_TYPES_H_
#define RECSYS_ENGINE_DATA_TYPES_H_

namespace recsys {
constexpr size_t EMBEDDING_DIM = 128;
constexpr size_t BATCH_SIZE = 256;
constexpr size_t NUM_NEON_PIPELINES = 4;

enum class InvalidArgmentError {
  EnumTypeUndefined,
  TypeDisagreesWithArenaType,
};

enum class EmbeddingDataType {
  Float64_t,
  Float32_t,
};

enum class RequestValidationError {
  RawDataSizeError,
  SearchLibraryNotFoundError,
  UnupportedDatatypeError
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

enum class ResponseStatus { StatusOk, StatusFailed };

} // namespace recsys
#endif // RECSYS_ENGINE_DATA_TYPES_H_