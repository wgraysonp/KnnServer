#ifndef RECSYS_ENGINE_SERVICE_PIPELINES_H_
#define RECSYS_ENGINE_SERVICE_PIPELINES_H_

#include <folly/Expected.h>

#include <memory>
#include <vector>

#include "src/arena.h"
#include "src/data/consts.h"
#include "src/data/gen-cpp2/data_types.h"
#include "src/service/gen-cpp2/service_types.h"
#include "src/service/service_utils.h"

namespace recsys::knn_server {

folly::Expected<std::unique_ptr<MemoryArena>, StartupError>
StartServiceAndCreateArena(const size_t item_count, const size_t embedding_dim,
                           const EmbeddingDataType& type);

folly::Expected<QueryResponse, SearchRequestError>
ProcessRequestAndReturnSearchResults(const MemoryArena& arena,
                                     const QueryRequest& request);

}  // namespace recsys::knn_server
#endif  // RECSYS_ENGINE_SERVICE_PIPELINES_H_