#include "src/service/pipelines.h"

#include <folly/Expected.h>

#include <memory>

#include "src/arena.h"
#include "src/data/structs.h"
#include "src/data/types.h"
#include "src/macros.h"
#include "src/service/service_utils.h"

namespace recsys {

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

folly::Expected<QueryResponse, SearchError>
ProcessRequestAndReturnSearchResults(const MemoryArena& arena,
                                     const QueryRequest& request) {
  // 1. validate request
  // 2. prepare response for search
  // 3. do the search
  // 4. package response and return it

  // temp placeholder to keep Werror Wunused-parameter from complaining
  if (arena.GetEmbeddingDim() != request.embedding_dim) {
    return folly::makeUnexpected(SearchError::InvalidQuery);
  }

  // temp placeholder
  return QueryResponse{ResponseStatus::StatusOk};
}

folly::Expected<folly::Unit, RequestValidationError> ValidateSearchRequest(
    const QueryRequest& request) {
  return ValidateRequestDataSize(request).then(
      [&request](
          folly::Unit) -> folly::Expected<folly::Unit, RequestValidationError> {
        return ValidateRequestSearchLibrary(request);
      });
}
}  // namespace recsys