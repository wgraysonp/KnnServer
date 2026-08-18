#include "src/service/pipelines.h"

#include <folly/Expected.h>

#include <memory>

#include "src/arena.h"
#include "src/consts.h"
#include "src/macros.h"

namespace recsys {

folly::Expected<std::unique_ptr<MemoryArena>, AllocError>
StartServiceAndCreateArena(size_t item_count, size_t embedding_dim,
                           EmbeddingDataType type) {
  return MemoryArena::MakeArena(item_count, embedding_dim, type)
      .then([item_count, embedding_dim](std::unique_ptr<MemoryArena> arena)
                -> folly::Expected<std::unique_ptr<MemoryArena>, AllocError> {
        for (size_t i = 0; i < item_count; ++i) {
          std::vector<float> test_vec(embedding_dim, i == 0 ? 0.5f : 0.3f);
          RETURN_IF_ERROR(arena->SetEntry(i, test_vec));
        }
        return arena;
      });
}

// TODO: the return type here should be something else: a collection of the
// closest vectors
folly::Expected<std::vector<float>, SearchError>
ProcessRequestAndReturnSearchResults();

}  // namespace recsys