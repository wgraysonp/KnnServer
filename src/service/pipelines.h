#ifndef RECSYS_ENGINE_SERVICE_PIPELINES_H_
#define RECSYS_ENGINE_SERVICE_PIPELINES_H_

#include <folly/Expected.h>

#include <memory>
#include <vector>

#include "src/arena.h"
#include "src/consts.h"
#include "src/data/structs.h"

namespace recsys {

enum class SearchError { InvalidQuery };

folly::Expected<std::unique_ptr<MemoryArena>, StartupError>
StartServiceAndCreateArena(size_t item_count, size_t embedding_dim,
                           EmbeddingDataType type);

// TODO: the return type here should be something else: a collection of the
// closest vectors
folly::Expected<std::vector<float>, SearchError>
ProcessRequestAndReturnSearchResults();

// Temporary implementation assuming the request consists only of a single
// float vector.
// TODO: This should be updated to handle a full http or protobuf request
folly::Expected<folly::Unit, SearchError> ValidateSearchRequest(
    const QueryRequest& request);

}  // namespace recsys
#endif  // RECSYS_ENGINE_SERVICE_PIPELINES_H_