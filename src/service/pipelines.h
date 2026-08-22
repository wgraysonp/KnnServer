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

folly::Expected<std::unique_ptr<MemoryArena>, StartupError>
StartServiceAndCreateArena(const size_t item_count, const size_t embedding_dim,
                           const EmbeddingDataType& type);

folly::Expected<QueryResponse, SearchRequestError>
ProcessRequestAndReturnSearchResults(const MemoryArena& arena,
                                     const QueryRequest& request);

}  // namespace recsys
#endif  // RECSYS_ENGINE_SERVICE_PIPELINES_H_