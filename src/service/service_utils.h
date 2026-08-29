#ifndef RECSYS_ENGINE_SERVICE_SERVICE_UTILS_H_
#define RECSYS_ENGINE_SERVICE_SERVICE_UTILS_H_

#include <folly/Expected.h>

#include <cstddef>
#include <cstdint>

#include "src/arena.h"
#include "src/data/consts.h"
#include "src/data/data_utils.h"
#include "src/data/gen-cpp2/data_types.h"
#include "src/data/structs.h"
#include "src/service/gen-cpp2/service_types.h"

namespace recsys::knn_server {

// Checks that the size of the raw bytes of the query vector sent in the
// request match the request datatype and embedding dimension.
folly::Expected<folly::Unit, SearchRequestError> ValidateRequestDataSize(
    const QueryRequest& request);

// Verifies that the requested library to search within exists and is valid.
folly::Expected<folly::Unit, SearchRequestError> ValidateRequestSearchLibrary(
    const QueryRequest& request);

// Temporary implementation assuming the request consists only of a single
// float vector.
// TODO: This should be updated to handle a full http or protobuf request
folly::Expected<folly::Unit, SearchRequestError> ValidateSearchRequest(
    const QueryRequest& request);

void PerformSearchAndPopulateResponse(const MemoryArena& arena,
                                      const QueryRequest& request,
                                      QueryResponse& mutable_response);

}  // namespace recsys::knn_server

#endif  // RECSYS_ENGINE_SERVICE_SERVICE_UTILS_H_