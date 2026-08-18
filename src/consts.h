#ifndef RECSYS_ENGINE_CONSTS_H_
#define RECSYS_ENGINE_CONSTS_H_

#include <cstddef>

namespace recsys {
  constexpr size_t EMBEDDING_DIM = 128;
  enum class EmbeddingDataType {Float64_t, Float32_t};
  enum class InvalidArgmentError{
    EnumTypeUndefined,
    TypeDisagreesWithArenaType,
  };
} // namespace recsys


#endif //RECSYS_ENGGINE_CONSTS_H_