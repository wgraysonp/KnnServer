#include "src/service/pipelines.h"

#include <folly/Expected.h>

#include <cstddef>
#include <memory>

#include "src/arena.h"
#include "src/data/consts.h"
#include "src/data/gen-cpp2/data_types.h"
#include "src/service/gen-cpp2/service_types.h"
#include "src/service/service_utils.h"

namespace recsys::knn_server {

folly::Expected<std::unique_ptr<MemoryArena>, StartupError>
StartServiceAndCreateArena(const size_t item_count, const size_t embedding_dim,
                           const EmbeddingDataType& type) {
  return MemoryArena::MakeArena(item_count, embedding_dim, type)
      .then([item_count, embedding_dim](std::unique_ptr<MemoryArena> arena)
                -> folly::Expected<std::unique_ptr<MemoryArena>, StartupError> {
        for (size_t i = 0; i < item_count; ++i) {
          std::vector<float> test_vec(embedding_dim, i == 3 ? 0.5f : 0.3f);
          if (!arena->SetEntry(i, test_vec)) {
            return folly::makeUnexpected(StartupError::Unknown);
          }
        }
        return arena;
      });
}

folly::Expected<folly::Unit, SearchRequestError>
ProcessRequestAndPopulateResponse(const MemoryArena& arena,
                                  const QueryRequest& request,
                                  QueryResponse& mutable_response) {
  return ValidateSearchRequest(request).then(
      [&arena, &request, &mutable_response](folly::Unit) {
        PerformSearchAndPopulateResponse(arena, request, mutable_response);
        return folly::unit;
      });
}
}  // namespace recsys::knn_server