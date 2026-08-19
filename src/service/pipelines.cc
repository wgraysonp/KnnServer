#include "src/service/pipelines.h"

#include <folly/Expected.h>

#include <memory>

#include "src/arena.h"
#include "src/consts.h"
#include "src/data/structs.h"
#include "src/macros.h"

namespace recsys {

folly::Expected<std::unique_ptr<MemoryArena>, StartupError>
StartServiceAndCreateArena(size_t item_count, size_t embedding_dim,
                           EmbeddingDataType type) {
  return MemoryArena::MakeArena(item_count, embedding_dim, type)
      .then([item_count, embedding_dim](std::unique_ptr<MemoryArena> arena)
                -> folly::Expected<std::unique_ptr<MemoryArena>, StartupError> {
        for (size_t i = 0; i < item_count; ++i) {
          std::vector<float> test_vec(embedding_dim, i == 0 ? 0.5f : 0.3f);
          if (!arena->SetEntry(i, test_vec)) {
            return folly::makeUnexpected(StartupError::Unknown);
          }
        }
        return arena;
      });
}

// TODO: the return type here should be something else: a collection of the
// closest vectors
// folly::Expected<std::vector<float>, SearchError>
// ProcessRequestAndReturnSearchResults();

// folly::Expected<folly::Unit, SearchError> ValidateSearchRequest(
//     const QueryRequest& request) {
//       auto 
//     }

}  // namespace recsys