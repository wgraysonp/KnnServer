#include "src/data/data_utils.h"

#include <cstddef>

#include "src/data/types.h"

namespace recsys {

size_t GetTypeSize(EmbeddingDataType type) {
  switch (type) {
    case EmbeddingDataType::Float32_t:
      return 4;
    case EmbeddingDataType::Float64_t:
      return 8;
  }
}

}  // namespace recsys