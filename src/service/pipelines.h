#ifndef RECSYS_ENGINE_SERVICE_PIPELINES_H_
#define RECSYS_ENGINE_SERVICE_PIPELINES_H_

#include <folly/Expected.h>

#include <memory>
#include <vector>

#include "src/arena.h"
#include "src/data/structs.h"
#include "src/data/types.h"
#include "src/service/service_utils.h"

namespace recsys {

enum class SearchError { InvalidQuery };

folly::Expected<std::unique_ptr<MemoryArena>, StartupError>
StartServiceAndCreateArena(const size_t item_count, const size_t embedding_dim,
                           const EmbeddingDataType type);

folly::Expected<QueryResponse, SearchError>
ProcessRequestAndReturnSearchResults(const MemoryArena& arena,
                                     const QueryRequest& request);

// Temporary implementation assuming the request consists only of a single
// float vector.
// TODO: This should be updated to handle a full http or protobuf request
folly::Expected<folly::Unit, RequestValidationError> ValidateSearchRequest(
    const QueryRequest& request);

}  // namespace recsys
#endif  // RECSYS_ENGINE_SERVICE_PIPELINES_H_