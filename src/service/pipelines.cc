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
          std::vector<float> test_vec(embedding_dim, i == 0 ? 0.5f : 0.3f);
          if (!arena->SetEntry(i, test_vec)) {
            return folly::makeUnexpected(StartupError::Unknown);
          }
        }
        return arena;
      });
}

folly::Expected<QueryResponse, SearchRequestError>
ProcessRequestAndReturnSearchResults(const MemoryArena& arena,
                                     const QueryRequest& request) {
  // 1. validate request
  // 2. prepare response for search
  // 3. do the search
  // 4. package response and return it

  // temp placeholder to keep Werror Wunused-parameter from complaining
  if (arena.GetEmbeddingDim() != request.embedding_dim_ref()) {
    return folly::makeUnexpected(SearchRequestError::InvalidRawDataSizeError);
  }

  // temp placeholder
  QueryResponse response;
  response.status_ref() = ResponseStatus::StatusOk;
  return response;
}
}  // namespace recsys::knn_server